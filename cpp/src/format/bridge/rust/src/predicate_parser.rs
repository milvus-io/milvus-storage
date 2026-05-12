// Copyright 2025 Zilliz
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

use anyhow::{bail, Result};
use sqlparser::ast::{BinaryOperator, Expr as SqlExpr, UnaryOperator, Value};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::Token;
use std::collections::HashMap;
use vortex::expr::{self, Expression};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    Int,
    UInt,
    Float,
    Utf8,
    Bool,
    Other,
    // Only set internally when no schema is supplied; literal type checks
    // are skipped against columns of this type.
    Unchecked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LitType {
    Int,
    Float,
    IntFloat,
    Utf8,
    Bool,
}

enum Out {
    Ok(Expression),
    OkCol { expr: Expression, col_ty: ColumnType },
    OkLit { expr: Expression, lit_ty: LitType },
    Drop,
}

pub fn parse_predicate(predicate: &str) -> Result<Option<Expression>> {
    parse_predicate_with_schema(predicate, &[])
}

pub fn parse_predicate_with_schema(
    predicate: &str,
    schema: &[(String, ColumnType)],
) -> Result<Option<Expression>> {
    let predicate = predicate.trim();
    if predicate.is_empty() {
        bail!("empty predicate");
    }
    let dialect = GenericDialect {};
    let mut parser = Parser::new(&dialect)
        .try_with_sql(predicate)
        .map_err(|e| anyhow::anyhow!("failed to tokenize predicate: {e}"))?;
    let sql_expr = parser
        .parse_expr()
        .map_err(|e| anyhow::anyhow!("failed to parse predicate: {e}"))?;
    // parse_expr stops at the first unconsumed token; reject trailing input.
    let next = parser.peek_token();
    if next.token != Token::EOF {
        bail!("unexpected trailing input after expression: {}", next.token);
    }

    let cols: HashMap<&str, ColumnType> =
        schema.iter().map(|(n, t)| (n.as_str(), *t)).collect();
    let schema_present = !schema.is_empty();
    let out = convert_expr(&sql_expr, &cols, schema_present)?;
    Ok(match out {
        Out::Ok(e) | Out::OkCol { expr: e, .. } | Out::OkLit { expr: e, .. } => Some(e),
        Out::Drop => None,
    })
}

fn warn(msg: impl AsRef<str>) {
    eprintln!("predicate: {}", msg.as_ref());
}

fn convert_expr(
    sql_expr: &SqlExpr,
    cols: &HashMap<&str, ColumnType>,
    schema_present: bool,
) -> Result<Out> {
    match sql_expr {
        SqlExpr::Identifier(ident) => {
            let name = ident.value.as_str();
            if !schema_present {
                return Ok(Out::OkCol {
                    expr: expr::get_item(ident.value.clone(), expr::root()),
                    col_ty: ColumnType::Unchecked,
                });
            }
            match cols.get(name) {
                Some(&ty) => Ok(Out::OkCol {
                    expr: expr::get_item(ident.value.clone(), expr::root()),
                    col_ty: ty,
                }),
                None => bail!("unknown column: {name}"),
            }
        }
        SqlExpr::Value(v) => Ok(convert_value(&v.value)),
        SqlExpr::UnaryOp { op, expr: operand } => match op {
            UnaryOperator::Not => {
                let inner = convert_expr(operand, cols, schema_present)?;
                Ok(match inner {
                    Out::Drop => Out::Drop,
                    other => match extract_expr(other) {
                        Some(e) => Out::Ok(expr::not(e)),
                        None => Out::Drop,
                    },
                })
            }
            UnaryOperator::Minus => match operand.as_ref() {
                SqlExpr::Value(v) => Ok(convert_negative_value(&v.value)),
                _ => {
                    warn("unsupported unary minus on non-literal expression");
                    Ok(Out::Drop)
                }
            },
            other => {
                warn(format!("unsupported unary operator: {other}"));
                Ok(Out::Drop)
            }
        },
        SqlExpr::BinaryOp { left, op, right } => {
            let lhs = convert_expr(left, cols, schema_present)?;
            let rhs = convert_expr(right, cols, schema_present)?;
            match op {
                BinaryOperator::And => Ok(combine_and(lhs, rhs)),
                BinaryOperator::Or => Ok(combine_or(lhs, rhs)),
                BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq
                | BinaryOperator::Lt
                | BinaryOperator::LtEq => Ok(combine_cmp(op.clone(), lhs, rhs)),
                other => {
                    warn(format!("unsupported binary operator: {other}"));
                    Ok(Out::Drop)
                }
            }
        }
        SqlExpr::Nested(inner) => convert_expr(inner, cols, schema_present),
        SqlExpr::InList { expr: target, list, negated } => {
            convert_in_list(target, list, *negated, cols, schema_present)
        }
        SqlExpr::Function(f) => {
            warn(format!("unsupported function call: {}", f.name));
            Ok(Out::Drop)
        }
        other => {
            warn(format!("unsupported expression: {other}"));
            Ok(Out::Drop)
        }
    }
}

fn convert_in_list(
    target: &SqlExpr,
    list: &[SqlExpr],
    negated: bool,
    cols: &HashMap<&str, ColumnType>,
    schema_present: bool,
) -> Result<Out> {
    let target_out = convert_expr(target, cols, schema_present)?;
    let (target_expr, target_col_ty) = match target_out {
        Out::OkCol { expr, col_ty } => (expr, Some(col_ty)),
        Out::Ok(expr) => (expr, None),
        Out::OkLit { .. } => {
            warn("IN target must be a column reference");
            return Ok(Out::Drop);
        }
        Out::Drop => return Ok(Out::Drop),
    };
    if list.is_empty() {
        warn("empty IN list");
        return Ok(Out::Drop);
    }
    let mut acc: Option<Expression> = None;
    for item in list {
        let item_out = convert_expr(item, cols, schema_present)?;
        let item_expr = match item_out {
            Out::OkLit { expr, lit_ty } => {
                if let Some(col_ty) = target_col_ty {
                    if !type_compatible(col_ty, lit_ty) {
                        warn(format!(
                            "type mismatch in IN list: column type {col_ty:?} vs literal type {lit_ty:?}"
                        ));
                        return Ok(Out::Drop);
                    }
                }
                expr
            }
            Out::Ok(e) | Out::OkCol { expr: e, .. } => e,
            Out::Drop => return Ok(Out::Drop),
        };
        let cmp = expr::eq(target_expr.clone(), item_expr);
        acc = Some(match acc {
            None => cmp,
            Some(prev) => expr::or(prev, cmp),
        });
    }
    let combined = acc.expect("non-empty IN list checked above");
    Ok(Out::Ok(if negated { expr::not(combined) } else { combined }))
}

fn combine_and(lhs: Out, rhs: Out) -> Out {
    match (extract_expr(lhs), extract_expr(rhs)) {
        (Some(l), Some(r)) => Out::Ok(expr::and(l, r)),
        (Some(l), None) => Out::Ok(l),
        (None, Some(r)) => Out::Ok(r),
        (None, None) => Out::Drop,
    }
}

fn combine_or(lhs: Out, rhs: Out) -> Out {
    match (extract_expr(lhs), extract_expr(rhs)) {
        (Some(l), Some(r)) => Out::Ok(expr::or(l, r)),
        _ => Out::Drop,
    }
}

fn combine_cmp(op: BinaryOperator, lhs: Out, rhs: Out) -> Out {
    let type_ok = match (&lhs, &rhs) {
        (Out::OkCol { col_ty, .. }, Out::OkLit { lit_ty, .. })
        | (Out::OkLit { lit_ty, .. }, Out::OkCol { col_ty, .. }) => {
            type_compatible(*col_ty, *lit_ty)
        }
        (Out::OkCol { col_ty: a, .. }, Out::OkCol { col_ty: b, .. }) => {
            column_to_column_compat(*a, *b)
        }
        _ => true,
    };
    if !type_ok {
        warn(format!("type mismatch in comparison: {op}"));
        return Out::Drop;
    }
    match (extract_expr(lhs), extract_expr(rhs)) {
        (Some(l), Some(r)) => match op {
            BinaryOperator::Eq => Out::Ok(expr::eq(l, r)),
            BinaryOperator::NotEq => Out::Ok(expr::not_eq(l, r)),
            BinaryOperator::Gt => Out::Ok(expr::gt(l, r)),
            BinaryOperator::GtEq => Out::Ok(expr::gt_eq(l, r)),
            BinaryOperator::Lt => Out::Ok(expr::lt(l, r)),
            BinaryOperator::LtEq => Out::Ok(expr::lt_eq(l, r)),
            _ => unreachable!("non-comparison operator routed to combine_cmp"),
        },
        _ => Out::Drop,
    }
}

fn extract_expr(o: Out) -> Option<Expression> {
    match o {
        Out::Ok(e) | Out::OkCol { expr: e, .. } | Out::OkLit { expr: e, .. } => Some(e),
        Out::Drop => None,
    }
}

fn type_compatible(col: ColumnType, lit: LitType) -> bool {
    match (col, lit) {
        (ColumnType::Unchecked, _) => true,
        (ColumnType::Int | ColumnType::UInt, LitType::Int | LitType::IntFloat) => true,
        (ColumnType::Float, LitType::Int | LitType::Float | LitType::IntFloat) => true,
        (ColumnType::Utf8, LitType::Utf8) => true,
        (ColumnType::Bool, LitType::Bool) => true,
        _ => false,
    }
}

fn column_to_column_compat(a: ColumnType, b: ColumnType) -> bool {
    if a == b {
        return true;
    }
    if matches!(a, ColumnType::Unchecked) || matches!(b, ColumnType::Unchecked) {
        return true;
    }
    matches!(
        (a, b),
        (ColumnType::Int, ColumnType::UInt)
            | (ColumnType::UInt, ColumnType::Int)
            | (ColumnType::Int, ColumnType::Float)
            | (ColumnType::Float, ColumnType::Int)
            | (ColumnType::UInt, ColumnType::Float)
            | (ColumnType::Float, ColumnType::UInt)
    )
}

fn convert_value(val: &Value) -> Out {
    match val {
        Value::Number(s, _) => {
            if let Ok(i) = s.parse::<i64>() {
                return Out::OkLit { expr: expr::lit(i), lit_ty: LitType::Int };
            }
            if let Ok(f) = s.parse::<f64>() {
                let lit_ty = if f.fract() == 0.0 {
                    LitType::IntFloat
                } else {
                    LitType::Float
                };
                return Out::OkLit { expr: expr::lit(f), lit_ty };
            }
            Out::Drop
        }
        Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => Out::OkLit {
            expr: expr::lit(s.as_str()),
            lit_ty: LitType::Utf8,
        },
        Value::Boolean(b) => Out::OkLit {
            expr: expr::lit(*b),
            lit_ty: LitType::Bool,
        },
        Value::Null => Out::Drop,
        _ => Out::Drop,
    }
}

fn convert_negative_value(val: &Value) -> Out {
    match val {
        Value::Number(s, _) => {
            if let Ok(i) = s.parse::<i64>() {
                return Out::OkLit {
                    expr: expr::lit(-i),
                    lit_ty: LitType::Int,
                };
            }
            if let Ok(f) = s.parse::<f64>() {
                let neg = -f;
                let lit_ty = if neg.fract() == 0.0 {
                    LitType::IntFloat
                } else {
                    LitType::Float
                };
                return Out::OkLit {
                    expr: expr::lit(neg),
                    lit_ty,
                };
            }
            Out::Drop
        }
        _ => Out::Drop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> Vec<(String, ColumnType)> {
        vec![
            ("age".into(), ColumnType::Int),
            ("name".into(), ColumnType::Utf8),
        ]
    }

    fn must_parse(predicate: &str) -> Expression {
        parse_predicate(predicate)
            .unwrap_or_else(|e| panic!("parse failed for '{predicate}': {e}"))
            .unwrap_or_else(|| panic!("parse yielded no filter for '{predicate}'"))
    }

    fn must_parse_with(predicate: &str, schema: &[(String, ColumnType)]) -> Expression {
        parse_predicate_with_schema(predicate, schema)
            .unwrap_or_else(|e| panic!("parse failed for '{predicate}': {e}"))
            .unwrap_or_else(|| panic!("parse yielded no filter for '{predicate}'"))
    }

    fn count_nodes(expr: &Expression) -> usize {
        let dbg = format!("{expr:?}");
        dbg.matches("vtable:").count()
    }

    // ---------- Original schema-less smoke tests ----------

    #[test]
    fn test_simple_comparison() {
        let expr = must_parse("age > 30");
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn test_all_digit_column() {
        let expr = must_parse("\"1\" > 10");
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn test_and_expression() {
        let expr = must_parse("a > 1 AND b < 2");
        assert_eq!(count_nodes(&expr), 9);
    }

    #[test]
    fn test_or_expression() {
        let expr = must_parse("a > 1 OR b < 2");
        assert_eq!(count_nodes(&expr), 9);
    }

    #[test]
    fn test_not_expression() {
        let expr = must_parse("NOT a > 1");
        assert_eq!(count_nodes(&expr), 5);
    }

    #[test]
    fn test_in_list_integers() {
        let expr = must_parse("status IN (1, 2, 3)");
        assert_eq!(count_nodes(&expr), 14);
    }

    #[test]
    fn test_in_list_strings() {
        let expr = must_parse("color IN ('red', 'blue', 'green')");
        assert_eq!(count_nodes(&expr), 14);
    }

    #[test]
    fn test_nested_parentheses() {
        let expr = must_parse("(a > 1 OR b > 2) AND c = 3");
        assert_eq!(count_nodes(&expr), 14);
    }

    #[test]
    fn test_negative_integer() {
        let expr = must_parse("x = -10");
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn test_negative_float() {
        let expr = must_parse("x > -3.14");
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn test_float_comparison() {
        let expr = must_parse("price >= 19.99");
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn test_operator_precedence() {
        let expr = must_parse("a = 1 OR b = 2 AND c = 3");
        assert_eq!(count_nodes(&expr), 14);
    }

    // ---------- New three-state behavior tests ----------

    #[test]
    fn unsupported_function_call_yields_drop() {
        // Schema-less: function call is unsupported -> filter is None.
        let r = parse_predicate("UPPER(name) = 'FOO'").unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn empty_predicate_is_err() {
        assert!(parse_predicate("").is_err());
        assert!(parse_predicate("   ").is_err());
    }

    #[test]
    fn err_on_syntax_error() {
        let schema = test_schema();
        let r = parse_predicate_with_schema("age >", &schema);
        assert!(r.is_err());
    }

    #[test]
    fn err_on_unknown_column() {
        let schema = test_schema();
        let r = parse_predicate_with_schema("missing > 1", &schema);
        assert!(r.is_err(), "unknown column must be Err");
    }

    #[test]
    fn err_on_unknown_column_inside_and() {
        let schema = test_schema();
        let r = parse_predicate_with_schema("age > 1 AND missing = 'x'", &schema);
        assert!(r.is_err(), "unknown column anywhere is hard error");
    }

    #[test]
    fn drop_unsupported_function_in_and() {
        let schema = test_schema();
        let expr = must_parse_with("age > 1 AND UPPER(name) = 'X'", &schema);
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn drop_type_mismatch_in_and() {
        let schema = test_schema();
        let expr = must_parse_with("age > 1 AND name = 5", &schema);
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn drop_propagates_to_whole_or() {
        let schema = test_schema();
        let r = parse_predicate_with_schema("age > 1 OR UPPER(name) = 'X'", &schema).unwrap();
        assert!(r.is_none(), "OR with Drop child becomes Drop");
    }

    #[test]
    fn drop_under_not_drops_whole() {
        let schema = test_schema();
        let r = parse_predicate_with_schema("NOT UPPER(name) = 'X'", &schema).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn integer_literal_compatible_with_float_column() {
        let schema = vec![("score".into(), ColumnType::Float)];
        let r = parse_predicate_with_schema("score > 5", &schema).unwrap();
        assert!(r.is_some());
    }

    #[test]
    fn float_with_fractional_part_drops_int_compare() {
        let schema = test_schema(); // age is Int
        let expr = must_parse_with("age = 1.5 AND age > 0", &schema);
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn legacy_no_schema_helper_still_works() {
        let r = parse_predicate("a > 1 AND b = 'x'").unwrap();
        assert!(r.is_some());
    }

    #[test]
    fn utf8_column_with_single_quoted_string_literal() {
        let schema = vec![("name".into(), ColumnType::Utf8)];
        let r = parse_predicate_with_schema("name = 'name_50'", &schema).unwrap();
        assert!(r.is_some());
    }

    // ---------- Edge cases on the SQL expression string ----------

    #[test]
    fn edge_unbalanced_paren_is_err() {
        let schema = test_schema();
        assert!(parse_predicate_with_schema("(age > 1", &schema).is_err());
    }

    #[test]
    fn edge_empty_in_list_drops() {
        let schema = test_schema();
        // sqlparser-rs may reject `IN ()` at parse time; treat either parse-err
        // or drop-with-no-filter as acceptable.
        match parse_predicate_with_schema("age IN ()", &schema) {
            Err(_) => {}
            Ok(None) => {}
            Ok(Some(_)) => panic!("empty IN list should not produce a usable filter"),
        }
    }

    #[test]
    fn edge_quoted_column_with_space_unknown() {
        let schema = test_schema();
        let r = parse_predicate_with_schema("\"my col\" > 1", &schema);
        assert!(r.is_err(), "unknown quoted column must be Err");
    }

    #[test]
    fn edge_literal_on_lhs() {
        let schema = test_schema();
        // `1 < age` — sqlparser accepts; both sides have a known shape; should
        // translate cleanly.
        let expr = must_parse_with("1 < age", &schema);
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn edge_double_not() {
        let schema = test_schema();
        let expr = must_parse_with("NOT NOT age > 1", &schema);
        // not(not(binary(gi(root), lit))) — 6 nodes
        assert_eq!(count_nodes(&expr), 6);
    }

    #[test]
    fn edge_column_eq_column_compatible() {
        let schema = vec![
            ("a".into(), ColumnType::Int),
            ("b".into(), ColumnType::Int),
        ];
        let expr = must_parse_with("a = b", &schema);
        assert_eq!(count_nodes(&expr), 5);
    }

    #[test]
    fn edge_sql_comment_in_predicate() {
        let schema = test_schema();
        // sqlparser strips `--` line comments. Should parse cleanly.
        let r = parse_predicate_with_schema("age > 1 -- ignored\n", &schema).unwrap();
        assert!(r.is_some());
    }

    #[test]
    fn edge_extra_statement_is_err() {
        // `1=1; DROP TABLE x` — parse_expr should stop after the first
        // expression and reject the rest as trailing tokens.
        let schema = test_schema();
        let r = parse_predicate_with_schema("1=1; DROP TABLE x", &schema);
        assert!(r.is_err(), "trailing statement must fail to parse");
    }

    #[test]
    fn edge_bool_column_with_bool_literal() {
        let schema = vec![("flag".into(), ColumnType::Bool)];
        let r = parse_predicate_with_schema("flag = true", &schema).unwrap();
        assert!(r.is_some());
    }

    #[test]
    fn edge_negated_in_list() {
        let schema = vec![("age".into(), ColumnType::Int)];
        let r = parse_predicate_with_schema("age NOT IN (1, 2)", &schema).unwrap();
        assert!(r.is_some());
    }

    #[test]
    fn edge_all_digit_column_schema_aware() {
        // Column names that look like integers must be addressable via quoting.
        let schema = vec![("1".into(), ColumnType::Int)];
        let expr = must_parse_with("\"1\" = 1", &schema);
        assert_eq!(count_nodes(&expr), 4);
    }
}
