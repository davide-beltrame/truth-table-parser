import sys
import re
import numpy as np
import os
import time

start = time.time()

def tokenize(input_text):
    tokens = []
    # Remove comments
    input_text = re.sub(r'#.*', '', input_text)
    token_specification = [
        ('SPECIAL', r'[();=]'),               # Special characters
        ('WORD', r'[A-Za-z_][A-Za-z0-9_]*'),  # Words
        ('SKIP', r'[ \t\r\n]+'),              # Blanks
        ('MISMATCH', r'.'),                   # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    for mo in re.finditer(tok_regex, input_text):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'WORD':
            tokens.append(value)
        elif kind == 'SPECIAL':
            tokens.append(value)
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise ValueError(f"Unexpected character '{value}'")
    return tokens

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else None
        self.definitions = {}  # Keep track of definitions to prevent redefinitions

    def error(self, message):
        raise Exception(f"Parser error at token '{self.current_token}': {message}")

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None

    def expect(self, expected_token):
        if self.current_token == expected_token:
            self.advance()
        else:
            self.error(f"Expected '{expected_token}'")

    def parse_program(self):
        instructions = []
        while self.current_token is not None:
            instr = self.parse_instruction()
            if instr is not None:
                instructions.append(instr)
        return instructions

    def parse_instruction(self):
        if self.current_token == 'var':
            self.advance()
            id_list = self.parse_id_list()
            if not id_list:
                self.error("Expected at least one variable after 'var'")
            for identifier in id_list:
                if identifier in self.definitions:
                    self.error(f"Identifier '{identifier}' already defined")
                self.definitions[identifier] = 'var'
            self.expect(';')
            return ('var', id_list)
        elif self.current_token == 'show':
            self.advance()
            id_list = self.parse_id_list()
            if not id_list:
                self.error("Expected at least one identifier after 'show'")
            self.expect(';')
            return ('show', id_list)
        elif self.current_token == 'show_ones':
            self.advance()
            id_list = self.parse_id_list()
            if not id_list:
                self.error("Expected at least one identifier after 'show_ones'")
            self.expect(';')
            return ('show_ones', id_list)
        elif self.is_identifier(self.current_token):
            identifier = self.current_token
            if identifier in self.definitions:
                self.error(f"Identifier '{identifier}' already defined")
            self.definitions[identifier] = 'assign'
            self.advance()
            self.expect('=')
            expr = self.parse_expr()
            self.expect(';')
            return ('assign', identifier, expr)
        else:
            self.error(f"Unexpected token '{self.current_token}'")

    def parse_id_list(self):
        id_list = []
        while self.current_token is not None and self.is_identifier(self.current_token):
            identifier = self.current_token
            id_list.append(identifier)
            self.advance()
        return id_list

    def parse_expr(self):
        if self.current_token == 'not':
            return self.parse_negation()
        else:
            return self.parse_conj_disj()

    def parse_conj_disj(self):
        expr = self.parse_paren_expr()
        while self.current_token == 'and' or self.current_token == 'or':
            op = self.current_token
            operands = [expr]
            while self.current_token == op:
                self.advance()
                next_expr = self.parse_paren_expr()
                operands.append(next_expr)
            if len(operands) < 2:
                self.error(f"'{op}' expression must have at least two operands")
            expr = (op, *operands)
        return expr

    def parse_negation(self):
        self.expect('not')
        expr = self.parse_paren_expr()
        return ('not', expr)

    def parse_paren_expr(self):
        if self.current_token == '(':
            self.advance()
            expr = self.parse_expr()
            self.expect(')')
            return expr
        elif self.is_element(self.current_token):
            return self.parse_element()
        else:
            self.error(f"Expected an element or '(' at '{self.current_token}'")

    def parse_element(self):
        if self.current_token == 'True':
            self.advance()
            return ('True',)
        elif self.current_token == 'False':
            self.advance()
            return ('False',)
        elif self.is_identifier(self.current_token):
            identifier = self.current_token
            if identifier not in self.definitions:
                self.error(f"Identifier '{identifier}' not defined")
            self.advance()
            return ('var', identifier)
        else:
            self.error(f"Unexpected token '{self.current_token}' in expression")

    def is_identifier(self, token):
        keywords = ['var', 'show', 'show_ones', 'not', 'and', 'or', 'True', 'False']
        return token not in keywords and re.match(r'[A-Za-z_][A-Za-z0-9_]*', token)

    def is_element(self, token):
        return token == 'True' or token == 'False' or self.is_identifier(token)

def generate_variable_arrays(variables):
    n = len(variables)
    if n > 64:
        raise Exception("Cannot handle more than 64 variables.")
    num_rows = 2 ** n
    indices = np.arange(num_rows, dtype=np.uint64)
    arrays = {}
    for i, var in enumerate(variables):
        arrays[var] = ((indices >> (n - i - 1)) & 1).astype(np.uint8)
    return arrays

def evaluate_expression(expr, variable_arrays, memo, assignments):
    if expr in memo:
        return memo[expr]
    op = expr[0]
    if op == 'and':
        results = []
        for sub_expr in expr[1:]:
            val = evaluate_expression(sub_expr, variable_arrays, memo, assignments)
            results.append(val)
        result = np.logical_and.reduce(results).astype(np.uint8)
    elif op == 'or':
        results = []
        for sub_expr in expr[1:]:
            val = evaluate_expression(sub_expr, variable_arrays, memo, assignments)
            results.append(val)
        result = np.logical_or.reduce(results).astype(np.uint8)
    elif op == 'not':
        val = evaluate_expression(expr[1], variable_arrays, memo, assignments)
        result = (~val) & 1
    elif op == 'var':
        var_name = expr[1]
        if var_name in variable_arrays:
            result = variable_arrays[var_name]
        elif var_name in memo:
            result = memo[var_name]
        elif var_name in assignments:
            assigned_expr = assignments[var_name]
            result = evaluate_expression(assigned_expr, variable_arrays, memo, assignments)
            memo[var_name] = result
        else:
            raise Exception(f"Variable '{var_name}' not found in variable arrays or assignments")
    elif op == 'True':
        num_rows = len(next(iter(variable_arrays.values())))
        result = np.ones(num_rows, dtype=np.uint8)
    elif op == 'False':
        num_rows = len(next(iter(variable_arrays.values())))
        result = np.zeros(num_rows, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown expression type: {op}")
    memo[expr] = result
    return result

def process_show_instruction(instr, variable_arrays, memo, variable_list, assignments):
    instr_type = instr[0]
    id_list = instr[1]
    num_rows = 2 ** len(variable_list)
    total_columns = len(variable_list) + len(id_list)
    output = np.zeros((num_rows, total_columns), dtype=np.uint8)
    for i, var in enumerate(variable_list):
        output[:, i] = variable_arrays[var]
    for i, identifier in enumerate(id_list):
        if identifier not in memo:
            result = evaluate_expression(('var', identifier), variable_arrays, memo, assignments)
            memo[identifier] = result
        output[:, len(variable_list) + i] = memo[identifier]
    header = '# ' + ' '.join(variable_list + id_list)
    print(header)
    if instr_type == 'show':
        for row in output:
            print('  ' + ' '.join(map(str, row)))
    elif instr_type == 'show_ones':
        selector = np.any(output[:, len(variable_list):], axis=1)
        selected_rows = output[selector]
        for row in selected_rows:
            print('  ' + ' '.join(map(str, row)))
    print('')

def process_file(input_file):
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return
    with open(input_file, 'r') as f:
        input_text = f.read()
    try:
        tokens = tokenize(input_text)
        parser = Parser(tokens)
        # Initialize variables and assignments
        variables = []
        assignments = {}
        instructions = parser.parse_program()
        memo = {}
        for instr in instructions:
            if instr[0] == 'var':
                # Update variables
                variables.extend(instr[1])
            elif instr[0] == 'assign':
                identifier = instr[1]
                expr = instr[2]
                assignments[identifier] = expr
            elif instr[0] == 'show' or instr[0] == 'show_ones':
                # Generate variable arrays using variables declared up to this point
                if len(variables) > 64:
                    raise Exception("Cannot handle more than 64 variables.")
                variable_arrays = generate_variable_arrays(variables)
                # Evaluate assignments made up to this point
                local_memo = {}
                # Copy assignments up to this point
                assignments_up_to_now = assignments.copy()
                # Evaluate only the necessary assignments
                for identifier in assignments_up_to_now:
                    if identifier not in local_memo:
                        result = evaluate_expression(('var', identifier), variable_arrays, local_memo, assignments_up_to_now)
                        local_memo[identifier] = result
                # Process the show instruction
                process_show_instruction(instr, variable_arrays, local_memo, variables, assignments_up_to_now)
            else:
                # Other instructions, do nothing
                pass
    except Exception as e:
        print(f"Error processing file '{input_file}': {e}")

def main():
    if len(sys.argv) > 1:
        # Process files provided as command-line arguments
        for input_file in sys.argv[1:]:
            print(f"# Processing {input_file}\n")
            process_file(input_file)
    else:
        time_list = []
        for x in range(0, 32):
            start_time = time.time()
            x_str = f'{x:02}'  # Format x as two-digit number with leading zeros
            filename = f'ag20_{x_str}.txt'
            input_file = f'hw01_instances/{filename}'
            print(f"# Processing {filename}\n")
            process_file(input_file)
            end_time = time.time()
            elapsed = end_time - start_time
            time_list.append(elapsed)
            print(f"{filename} took {elapsed:.4f} seconds")
        # Print summary of times
        print("\n# Summary of times:")
        for i, elapsed in enumerate(time_list):
            x_str = f'{i:02}'
            print(f"# ag20_{x_str}.txt took {elapsed:.4f} seconds")

if __name__ == '__main__':
    main()
    end_time = time.time()
    total_time = end_time - start
    print(f"# Total time: {total_time:.4f} seconds")