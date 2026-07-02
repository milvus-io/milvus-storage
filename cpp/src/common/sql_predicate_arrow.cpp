// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common/sql_predicate_arrow.h"

#include <cctype>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/compute/expression.h>

// A self-contained, lightweight predicate parser.
//
// It deliberately avoids pulling in a full SQL parser (e.g. Hyrise) because the
// predicate-delete payload only needs a small boolean expression grammar. The
// whole front-end (tokenizer + recursive-descent parser) lives in this single
// file and emits `arrow::compute::Expression` directly, reusing Arrow for
// type/null semantics during Bind/Execute.
//
// Supported grammar (case-insensitive keywords):
//   or_expr    := and_expr ( "or" and_expr )*
//   and_expr   := not_expr ( "and" not_expr )*
//   not_expr   := "not" not_expr | comparison
//   comparison := primary ( cmp_op primary
//                          | "is" [ "not" ] "null"
//                          | "in" "(" literal_list ")" )?
//   primary    := "(" or_expr ")" | column_ref | literal
//   cmp_op     := "=" | "!=" | "<>" | "<" | "<=" | ">" | ">="
//   literal    := int | float | string | "true" | "false"
//
// Anything outside this grammar (functions, casts, LIKE, BETWEEN, arithmetic,
// subqueries, qualified columns, dynamic-field markers, placeholders, bare NULL
// literals) is rejected — illegal characters are rejected by the tokenizer, so
// arithmetic/`$meta`/`?`/qualified `t.a` need no dedicated parser rules.

namespace milvus_storage {
namespace {

namespace cp = arrow::compute;

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

enum class TokenKind {
  // literals / identifiers
  kIdentifier,
  kInt,
  kFloat,
  kString,
  kBool,
  // keywords
  kAnd,
  kOr,
  kNot,
  kIn,
  kIs,
  kNull,
  kLike,
  kBetween,
  // operators / punctuation
  kEq,
  kNe,
  kLt,
  kLe,
  kGt,
  kGe,
  kLParen,
  kRParen,
  kComma,
  kEnd,
};

struct Token {
  TokenKind kind;
  std::string text;  // identifier name or string-literal value
  int64_t int_value = 0;
  double float_value = 0;
  bool bool_value = false;
};

std::string ToLowerAscii(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (char c : input) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

class Tokenizer {
  public:
  explicit Tokenizer(const std::string& input) : input_(input) {}

  arrow::Result<std::vector<Token>> Tokenize() {
    std::vector<Token> tokens;
    while (pos_ < input_.size()) {
      char c = input_[pos_];
      if (std::isspace(static_cast<unsigned char>(c))) {
        ++pos_;
        continue;
      }
      if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        tokens.push_back(ReadIdentifierOrKeyword());
      } else if (std::isdigit(static_cast<unsigned char>(c)) ||
                 (c == '-' && pos_ + 1 < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_ + 1])))) {
        // A leading '-' immediately followed by a digit is a negative numeric
        // literal (e.g. `col < -5`). A '-' not followed by a digit (e.g. `-a`)
        // still falls through to ReadOperator and is rejected, so unary minus
        // stays unsupported.
        ARROW_ASSIGN_OR_RAISE(auto token, ReadNumber());
        tokens.push_back(std::move(token));
      } else if (c == '\'') {
        ARROW_ASSIGN_OR_RAISE(auto token, ReadString());
        tokens.push_back(std::move(token));
      } else {
        ARROW_ASSIGN_OR_RAISE(auto token, ReadOperator());
        tokens.push_back(std::move(token));
      }
    }
    tokens.push_back(Token{TokenKind::kEnd});
    return tokens;
  }

  private:
  Token ReadIdentifierOrKeyword() {
    size_t start = pos_;
    while (pos_ < input_.size() && (std::isalnum(static_cast<unsigned char>(input_[pos_])) || input_[pos_] == '_')) {
      ++pos_;
    }
    std::string text = input_.substr(start, pos_ - start);
    const std::string lower = ToLowerAscii(text);
    if (lower == "and")
      return Token{TokenKind::kAnd};
    if (lower == "or")
      return Token{TokenKind::kOr};
    if (lower == "not")
      return Token{TokenKind::kNot};
    if (lower == "in")
      return Token{TokenKind::kIn};
    if (lower == "is")
      return Token{TokenKind::kIs};
    if (lower == "null")
      return Token{TokenKind::kNull};
    if (lower == "like" || lower == "ilike")
      return Token{TokenKind::kLike};
    if (lower == "between")
      return Token{TokenKind::kBetween};
    if (lower == "true") {
      Token token{TokenKind::kBool};
      token.bool_value = true;
      return token;
    }
    if (lower == "false") {
      Token token{TokenKind::kBool};
      token.bool_value = false;
      return token;
    }
    // Preserve original case so column lookups stay case-sensitive.
    Token token{TokenKind::kIdentifier};
    token.text = std::move(text);
    return token;
  }

  arrow::Result<Token> ReadNumber() {
    size_t start = pos_;
    if (input_[pos_] == '-') {
      ++pos_;  // consume the leading minus of a negative literal (dispatch guarantees a digit follows)
    }
    while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
      ++pos_;
    }
    bool is_float = false;
    if (pos_ < input_.size() && input_[pos_] == '.') {
      is_float = true;
      ++pos_;
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
        ++pos_;
      }
    }
    const std::string text = input_.substr(start, pos_ - start);
    try {
      if (is_float) {
        Token token{TokenKind::kFloat};
        token.float_value = std::stod(text);
        return token;
      }
      Token token{TokenKind::kInt};
      token.int_value = static_cast<int64_t>(std::stoll(text));
      return token;
    } catch (const std::exception&) {
      return arrow::Status::Invalid("Invalid numeric literal in predicate SQL: ", text);
    }
  }

  arrow::Result<Token> ReadString() {
    ++pos_;  // consume opening quote
    std::string value;
    while (pos_ < input_.size()) {
      char c = input_[pos_];
      if (c == '\'') {
        // Two consecutive quotes ('') represent an escaped single quote.
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '\'') {
          value.push_back('\'');
          pos_ += 2;
          continue;
        }
        ++pos_;  // consume closing quote
        Token token{TokenKind::kString};
        token.text = std::move(value);
        return token;
      }
      value.push_back(c);
      ++pos_;
    }
    return arrow::Status::Invalid("Unterminated string literal in predicate SQL");
  }

  arrow::Result<Token> ReadOperator() {
    char c = input_[pos_];
    switch (c) {
      case '=':
        ++pos_;
        return Token{TokenKind::kEq};
      case '!':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          pos_ += 2;
          return Token{TokenKind::kNe};
        }
        return arrow::Status::Invalid("Unexpected character '!' in predicate SQL");
      case '<':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          pos_ += 2;
          return Token{TokenKind::kLe};
        }
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '>') {
          pos_ += 2;
          return Token{TokenKind::kNe};
        }
        ++pos_;
        return Token{TokenKind::kLt};
      case '>':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          pos_ += 2;
          return Token{TokenKind::kGe};
        }
        ++pos_;
        return Token{TokenKind::kGt};
      case '(':
        ++pos_;
        return Token{TokenKind::kLParen};
      case ')':
        ++pos_;
        return Token{TokenKind::kRParen};
      case ',':
        ++pos_;
        return Token{TokenKind::kComma};
      default:
        return arrow::Status::Invalid("Unexpected character '", std::string(1, c), "' in predicate SQL");
    }
  }

  const std::string& input_;
  size_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// Recursive-descent parser: tokens -> arrow::compute::Expression
// ---------------------------------------------------------------------------

class Parser {
  public:
  Parser(const std::vector<Token>& tokens, const std::shared_ptr<arrow::Schema>& schema)
      : tokens_(tokens), schema_(schema) {}

  arrow::Result<cp::Expression> Parse() {
    ARROW_ASSIGN_OR_RAISE(auto expr, ParseOr());
    if (Peek().kind != TokenKind::kEnd) {
      return arrow::Status::Invalid("Unexpected trailing tokens in predicate SQL");
    }
    return expr;
  }

  private:
  const Token& Peek() const { return tokens_[pos_]; }
  void Advance() { ++pos_; }
  bool Match(TokenKind kind) {
    if (tokens_[pos_].kind == kind) {
      ++pos_;
      return true;
    }
    return false;
  }

  arrow::Result<cp::Expression> ParseOr() {
    ARROW_ASSIGN_OR_RAISE(auto expr, ParseAnd());
    while (Match(TokenKind::kOr)) {
      ARROW_ASSIGN_OR_RAISE(auto rhs, ParseAnd());
      expr = cp::or_(std::move(expr), std::move(rhs));
    }
    return expr;
  }

  arrow::Result<cp::Expression> ParseAnd() {
    ARROW_ASSIGN_OR_RAISE(auto expr, ParseNot());
    while (Match(TokenKind::kAnd)) {
      ARROW_ASSIGN_OR_RAISE(auto rhs, ParseNot());
      expr = cp::and_(std::move(expr), std::move(rhs));
    }
    return expr;
  }

  arrow::Result<cp::Expression> ParseNot() {
    if (Match(TokenKind::kNot)) {
      ARROW_ASSIGN_OR_RAISE(auto child, ParseNot());
      return cp::not_(std::move(child));
    }
    return ParseComparison();
  }

  arrow::Result<cp::Expression> ParseComparison() {
    ARROW_ASSIGN_OR_RAISE(auto lhs, ParsePrimary());
    switch (Peek().kind) {
      case TokenKind::kEq:
      case TokenKind::kNe:
      case TokenKind::kLt:
      case TokenKind::kLe:
      case TokenKind::kGt:
      case TokenKind::kGe: {
        TokenKind op = Peek().kind;
        Advance();
        if (Peek().kind == TokenKind::kNull) {
          return arrow::Status::NotImplemented("NULL literal is only supported through IS NULL in predicate SQL");
        }
        ARROW_ASSIGN_OR_RAISE(auto rhs, ParsePrimary());
        return MakeComparison(op, std::move(lhs), std::move(rhs));
      }
      case TokenKind::kIs: {
        Advance();
        bool negated = Match(TokenKind::kNot);
        if (!Match(TokenKind::kNull)) {
          return arrow::Status::Invalid("Expected NULL after IS in predicate SQL");
        }
        cp::Expression expr = cp::is_null(std::move(lhs));
        return negated ? cp::not_(std::move(expr)) : expr;
      }
      case TokenKind::kIn:
        Advance();
        return ParseInList(std::move(lhs));
      case TokenKind::kLike:
        return arrow::Status::NotImplemented("LIKE operators are not supported in predicate SQL");
      case TokenKind::kBetween:
        return arrow::Status::NotImplemented("BETWEEN is not supported in predicate SQL");
      default:
        // Bare term (e.g. a boolean column) is allowed; Arrow validates its type.
        return lhs;
    }
  }

  arrow::Result<cp::Expression> ParseInList(cp::Expression lhs) {
    if (!Match(TokenKind::kLParen)) {
      return arrow::Status::Invalid("Expected '(' after IN in predicate SQL");
    }
    if (Peek().kind == TokenKind::kRParen) {
      return arrow::Status::Invalid("SQL IN operator expects a non-empty value list");
    }
    std::vector<cp::Expression> matches;
    while (true) {
      if (Peek().kind == TokenKind::kNull) {
        return arrow::Status::NotImplemented("NULL values in SQL IN lists are not supported in predicate SQL");
      }
      // IN lists only accept literals (no columns / subqueries).
      ARROW_ASSIGN_OR_RAISE(auto value, ParseLiteral());
      matches.push_back(cp::equal(lhs, std::move(value)));
      if (Match(TokenKind::kComma)) {
        continue;
      }
      break;
    }
    if (!Match(TokenKind::kRParen)) {
      return arrow::Status::Invalid("Expected ')' to close IN value list in predicate SQL");
    }
    cp::Expression result = std::move(matches.front());
    for (size_t i = 1; i < matches.size(); ++i) {
      result = cp::or_(std::move(result), std::move(matches[i]));
    }
    return result;
  }

  arrow::Result<cp::Expression> ParsePrimary() {
    const Token& token = Peek();
    if (token.kind == TokenKind::kLParen) {
      Advance();
      ARROW_ASSIGN_OR_RAISE(auto expr, ParseOr());
      if (!Match(TokenKind::kRParen)) {
        return arrow::Status::Invalid("Expected ')' in predicate SQL");
      }
      return expr;
    }
    if (token.kind == TokenKind::kIdentifier) {
      return ParseColumnRef();
    }
    if (token.kind == TokenKind::kNull) {
      return arrow::Status::NotImplemented("NULL literal is only supported through IS NULL in predicate SQL");
    }
    return ParseLiteral();
  }

  arrow::Result<cp::Expression> ParseColumnRef() {
    const std::string name = Peek().text;
    Advance();
    if (schema_->GetFieldIndex(name) < 0) {
      return arrow::Status::Invalid("Unknown column in predicate SQL: ", name);
    }
    return cp::field_ref(name);
  }

  arrow::Result<cp::Expression> ParseLiteral() {
    const Token& token = Peek();
    switch (token.kind) {
      case TokenKind::kInt: {
        int64_t value = token.int_value;
        Advance();
        return cp::literal(value);
      }
      case TokenKind::kFloat: {
        double value = token.float_value;
        Advance();
        return cp::literal(value);
      }
      case TokenKind::kString: {
        std::string value = token.text;
        Advance();
        return cp::literal(std::move(value));
      }
      case TokenKind::kBool: {
        bool value = token.bool_value;
        Advance();
        return cp::literal(value);
      }
      default:
        return arrow::Status::Invalid("Expected a literal or column in predicate SQL");
    }
  }

  static arrow::Result<cp::Expression> MakeComparison(TokenKind op, cp::Expression lhs, cp::Expression rhs) {
    switch (op) {
      case TokenKind::kEq:
        return cp::equal(std::move(lhs), std::move(rhs));
      case TokenKind::kNe:
        return cp::not_equal(std::move(lhs), std::move(rhs));
      case TokenKind::kLt:
        return cp::less(std::move(lhs), std::move(rhs));
      case TokenKind::kLe:
        return cp::less_equal(std::move(lhs), std::move(rhs));
      case TokenKind::kGt:
        return cp::greater(std::move(lhs), std::move(rhs));
      case TokenKind::kGe:
        return cp::greater_equal(std::move(lhs), std::move(rhs));
      default:
        return arrow::Status::Invalid("Unsupported comparison operator in predicate SQL");
    }
  }

  const std::vector<Token>& tokens_;
  const std::shared_ptr<arrow::Schema>& schema_;
  size_t pos_ = 0;
};

}  // namespace

arrow::Result<cp::Expression> ParseSqlPredicateToArrowExpression(const std::string& predicate_sql,
                                                                 const std::shared_ptr<arrow::Schema>& schema) {
  if (!schema) {
    return arrow::Status::Invalid("Arrow schema is required for SQL predicate conversion");
  }
  if (predicate_sql.empty()) {
    return arrow::Status::Invalid("Predicate SQL must not be empty");
  }

  ARROW_ASSIGN_OR_RAISE(auto tokens, Tokenizer(predicate_sql).Tokenize());
  Parser parser(tokens, schema);
  return parser.Parse();
}

}  // namespace milvus_storage
