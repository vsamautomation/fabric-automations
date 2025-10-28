#!/usr/bin/env python3
"""
CLI tool for converting M code to SQL
Usage: python m_to_sql_cli.py --input mcode.txt --output query.sql --table source_table
"""

import argparse
import sys
from pathlib import Path
from m_to_sql_transformer import transform_m_to_sql


def main():
    parser = argparse.ArgumentParser(
        description='Convert Power Query M code to SQL SELECT statements'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input file containing M code (or use stdin if not provided)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for SQL (or use stdout if not provided)'
    )
    parser.add_argument(
        '-t', '--table',
        type=str,
        default='source_table',
        help='Name of the source table in SQL (default: source_table)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show transformation steps'
    )
    
    args = parser.parse_args()
    
    # Read M code
    if args.input:
        with open(args.input, 'r') as f:
            m_code = f.read()
    else:
        print("Enter M code (Ctrl+D when done):", file=sys.stderr)
        m_code = sys.stdin.read()
    
    # Transform
    try:
        sql, steps = transform_m_to_sql(m_code, args.table)
        
        # Output SQL
        if args.output:
            with open(args.output, 'w') as f:
                f.write(sql)
            print(f"SQL written to {args.output}", file=sys.stderr)
        else:
            print(sql)
        
        # Show steps if verbose
        if args.verbose:
            print("\n" + "="*60, file=sys.stderr)
            print(f"Parsed {len(steps)} transformation steps:", file=sys.stderr)
            for i, step in enumerate(steps, 1):
                print(f"\n{i}. {step.step_name}", file=sys.stderr)
                print(f"   Operation: {step.operation}", file=sys.stderr)
                if step.parameters:
                    print(f"   Parameters: {step.parameters}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
