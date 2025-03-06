# **Truth Table Parser**

This repository contains a solution for the first assignment in the **20875 Software Engineering** course. The goal of this project is to **parse and evaluate logical expressions** defined in a simple custom programming language and generate **truth tables**.

## **Features**
- Tokenizes and parses a minimalistic logic-based language.
- Supports variable definitions, assignments, and logical operations (`not`, `and`, `or`).
- Generates **truth tables** from the defined expressions.
- Handles commands like:
  - `var x y;` (declares variables)
  - `x = not y;` (assigns logical expressions)
  - `show x y;` (displays a truth table)
  - `show_ones x;` (shows only rows where `x` is true)
- Processes multiple input files stored in `hw01_instances/`.

---

## **Usage**
To run the script on input files:
```bash
python3 table.py hw01_instances/example.txt
```
Or process all predefined instances:
```bash
python3 table.py
```

**Example input (`example.txt`):**
```
var x y;
a = not x;
b = x and y;
show a b;
```

**Expected output:**
```
# x y a b
  0 0 1 0
  0 1 1 0
  1 0 0 0
  1 1 0 1
```

---

## **Dependencies**
```bash
pip install numpy
```

---

## **Implementation Details**
- **Tokenizer**: Extracts keywords, variables, and operators.
- **Parser**: Constructs an abstract syntax tree (AST) for expressions.
- **Evaluator**: Computes Boolean values for all possible input combinations.
- **Truth Table Generator**: Displays results in a structured format.
