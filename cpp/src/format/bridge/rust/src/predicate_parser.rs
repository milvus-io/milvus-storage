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

//! Parses a SQL WHERE-clause predicate string into a `vortex::expr::Expression`.
//!
//! The predicate is wrapped in `SELECT * FROM t WHERE <predicate>` so that
//! `sqlparser` can parse it as valid SQL. The WHERE clause AST is then
//! converted into a Vortex expression tree.

use anyhow::{bail, Result};
use sqlparser::ast::{
    BinaryOperator, Expr as SqlExpr, SetExpr, Statement, UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use vortex::expr::{self, Expression};

/// Parse a SQL predicate string into a Vortex `Expression`.
///
/// The predicate should be a valid SQL WHERE-clause expression, e.g.:
/// - `"age > 30"`
/// - `"status IN (1, 2, 3)"`
/// - `"(a > 1 OR b > 2) AND c = 3"`
pub fn parse_predicate(predicate: &str) -> Result<Expression> {
    let predicate = predicate.trim();
    if predicate.is_empty() {
        bail!("empty predicate");
    }

    let sql = format!("SELECT * FROM t WHERE {predicate}");
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, &sql)
        .map_err(|e| anyhow::anyhow!("failed to parse predicate: {e}"))?;

    let statement = statements
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no statement parsed"))?;

    let where_expr = extract_where(statement)?;
    convert_expr(&where_expr)
}

/// Extract the WHERE clause expression from a parsed SELECT statement.
fn extract_where(stmt: Statement) -> Result<SqlExpr> {
    match stmt {
        Statement::Query(query) => match *query.body {
            SetExpr::Select(select) => select
                .selection
                .ok_or_else(|| anyhow::anyhow!("no WHERE clause found")),
            _ => bail!("unexpected query body"),
        },
        _ => bail!("unexpected statement type"),
    }
}

/// Convert a sqlparser AST expression into a Vortex expression.
fn convert_expr(sql_expr: &SqlExpr) -> Result<Expression> {
    match sql_expr {
        // Column reference: bare or quoted identifier
        SqlExpr::Identifier(ident) => {
            Ok(expr::get_item(ident.value.clone(), expr::root()))
        }

        // Literal values
        SqlExpr::Value(val) => convert_value(&val.value),

        // Unary operators: NOT, negative sign
        SqlExpr::UnaryOp { op, expr: operand } => match op {
            UnaryOperator::Not => {
                let inner = convert_expr(operand)?;
                Ok(expr::not(inner))
            }
            UnaryOperator::Minus => {
                // Negative literal: apply sign to the inner value
                match operand.as_ref() {
                    SqlExpr::Value(val) => convert_negative_value(&val.value),
                    _ => bail!("unsupported unary minus on non-literal expression"),
                }
            }
            _ => bail!("unsupported unary operator: {op}"),
        },

        // Binary operators: comparison, logical
        SqlExpr::BinaryOp { left, op, right } => {
            let lhs = convert_expr(left)?;
            let rhs = convert_expr(right)?;
            match op {
                BinaryOperator::Eq => Ok(expr::eq(lhs, rhs)),
                BinaryOperator::NotEq => Ok(expr::not_eq(lhs, rhs)),
                BinaryOperator::Gt => Ok(expr::gt(lhs, rhs)),
                BinaryOperator::GtEq => Ok(expr::gt_eq(lhs, rhs)),
                BinaryOperator::Lt => Ok(expr::lt(lhs, rhs)),
                BinaryOperator::LtEq => Ok(expr::lt_eq(lhs, rhs)),
                BinaryOperator::And => Ok(expr::and(lhs, rhs)),
                BinaryOperator::Or => Ok(expr::or(lhs, rhs)),
                _ => bail!("unsupported binary operator: {op}"),
            }
        }

        // Parenthesized expressions: (expr)
        SqlExpr::Nested(inner) => convert_expr(inner),

        // IN list: column IN (1, 2, 3)
        SqlExpr::InList {
            expr: target,
            list,
            negated,
        } => {
            let col = convert_expr(target)?;
            if list.is_empty() {
                bail!("empty IN list");
            }
            // Build: col = v1 OR col = v2 OR ...
            let mut result: Option<Expression> = None;
            for item in list {
                let val = convert_expr(item)?;
                let cmp = expr::eq(col.clone(), val);
                result = Some(match result {
                    None => cmp,
                    Some(acc) => expr::or(acc, cmp),
                });
            }
            let result = result.unwrap();
            if *negated {
                Ok(expr::not(result))
            } else {
                Ok(result)
            }
        }

        // Function calls are not supported
        SqlExpr::Function(f) => {
            bail!("unsupported expression: function call `{}`", f.name)
        }

        _ => bail!("unsupported expression: {sql_expr}"),
    }
}

/// Convert a sqlparser Value to a Vortex literal expression.
fn convert_value(val: &Value) -> Result<Expression> {
    match val {
        Value::Number(s, _is_long) => {
            // Try integer first, then float
            if let Ok(i) = s.parse::<i64>() {
                Ok(expr::lit(i))
            } else if let Ok(f) = s.parse::<f64>() {
                Ok(expr::lit(f))
            } else {
                bail!("cannot parse number: {s}")
            }
        }
        Value::SingleQuotedString(s) => Ok(expr::lit(s.as_str())),
        Value::DoubleQuotedString(s) => Ok(expr::lit(s.as_str())),
        Value::Boolean(b) => Ok(expr::lit(*b)),
        Value::Null => {
            // Represent NULL as is_null on root — but this is unusual in
            // predicates. For now, bail.
            bail!("NULL literal not supported in predicates")
        }
        _ => bail!("unsupported literal value: {val}"),
    }
}

/// Convert a negated sqlparser Value to a Vortex literal expression.
fn convert_negative_value(val: &Value) -> Result<Expression> {
    match val {
        Value::Number(s, _) => {
            if let Ok(i) = s.parse::<i64>() {
                Ok(expr::lit(-i))
            } else if let Ok(f) = s.parse::<f64>() {
                Ok(expr::lit(-f))
            } else {
                bail!("cannot parse negative number: -{s}")
            }
        }
        _ => bail!("unsupported negative literal: -{val}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: parse must succeed.
    fn must_parse(predicate: &str) -> Expression {
        parse_predicate(predicate).unwrap_or_else(|e| panic!("parse failed for '{predicate}': {e}"))
    }

    /// Count expression tree nodes by examining Debug output for "vtable:" occurrences.
    fn count_nodes(expr: &Expression) -> usize {
        let dbg = format!("{expr:?}");
        dbg.matches("vtable:").count()
    }

    #[test]
    fn test_simple_comparison() {
        // age > 30 => binary(get_item(root), literal) — 4 nodes
        let expr = must_parse("age > 30");
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn test_all_digit_column() {
        // "1" > 10 => binary(get_item(root), literal) — 4 nodes
        let expr = must_parse("\"1\" > 10");
        assert_eq!(count_nodes(&expr), 4);
    }

    #[test]
    fn test_and_expression() {
        // a > 1 AND b < 2 => binary(binary(gi(root),lit), binary(gi(root),lit)) — 9 nodes
        let expr = must_parse("a > 1 AND b < 2");
        assert_eq!(count_nodes(&expr), 9);
    }

    #[test]
    fn test_or_expression() {
        // a > 1 OR b < 2 => same structure as AND
        let expr = must_parse("a > 1 OR b < 2");
        assert_eq!(count_nodes(&expr), 9);
    }

    #[test]
    fn test_not_expression() {
        // NOT a > 1 => not(binary(gi,lit)) — 5 nodes
        let expr = must_parse("NOT a > 1");
        assert_eq!(count_nodes(&expr), 5);
    }

    #[test]
    fn test_in_list_integers() {
        // status IN (1, 2, 3) => or(or(eq(gi,lit), eq(gi,lit)), eq(gi,lit))
        // Each eq has gi(root)+lit = 4 nodes, 3 eqs = 12, plus 2 or = 14
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
        // (a > 1 OR b > 2) AND c = 3
        // or(gt(gi(root),lit), gt(gi(root),lit)) = 1+4+4 = 9
        // eq(gi(root),lit) = 4
        // and(or_tree, eq_tree) = 1+9+4 = 14
        let expr = must_parse("(a > 1 OR b > 2) AND c = 3");
        assert_eq!(count_nodes(&expr), 14);
    }

    #[test]
    fn test_negative_integer() {
        // x = -10 => binary(get_item(root), literal) — 4 nodes
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
        // a = 1 OR b = 2 AND c = 3
        // SQL precedence: AND binds tighter => a = 1 OR (b = 2 AND c = 3)
        // Top-level is OR with 2 children: eq(a,1) and and(eq(b,2),eq(c,3))
        // eq=4 nodes each, and=1+4+4=9, or=1+4+9=14 — but "and" wraps two eqs
        // Actually: or( eq(gi(root),lit), and(eq(gi(root),lit), eq(gi(root),lit)) )
        // = 1(or) + 4(eq) + 1(and) + 4(eq) + 4(eq) = 14
        let _expr = must_parse("a = 1 OR b = 2 AND c = 3");
        // Just verify it parses; the structure is correct if sqlparser handles
        // standard SQL operator precedence.
        assert_eq!(count_nodes(&_expr), 14);
    }

    #[test]
    fn test_unsupported_function_call() {
        let result = parse_predicate("UPPER(name) = 'FOO'");
        assert!(result.is_err(), "function call should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("function") || err.contains("unsupported"),
            "error should mention function: {err}"
        );
    }

    #[test]
    fn test_empty_predicate() {
        let result = parse_predicate("");
        assert!(result.is_err(), "empty predicate should fail");
        let result2 = parse_predicate("   ");
        assert!(result2.is_err(), "whitespace-only predicate should fail");
    }
}
