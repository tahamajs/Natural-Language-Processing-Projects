# Automatic Differentiation System: Backpropagation Implementation

**Course**: Natural Language Processing  
**Institution**: ETH Zürich / Sharif University of Technology  
**Assignment**: Assignment 1 - Backpropagation  
**Academic Term**: Autumn 2022

---

## Abstract

This project implements a complete automatic differentiation system capable of computing both function values and their gradients through reverse-mode differentiation (backpropagation). The system constructs computation graphs from mathematical expressions in infix notation and efficiently evaluates derivatives using the chain rule. The implementation supports a comprehensive set of mathematical operations including transcendental functions (exponential, logarithmic, trigonometric) and standard arithmetic operations.

**Keywords**: Automatic Differentiation, Backpropagation, Computation Graphs, Chain Rule, Gradient Computation

---

## I. Introduction

### A. Motivation

Automatic differentiation is fundamental to modern machine learning, particularly in training neural networks where gradients must be computed efficiently for millions of parameters. Unlike numerical differentiation (which suffers from truncation errors) or symbolic differentiation (which can produce inefficient expressions), automatic differentiation provides exact derivatives while maintaining computational efficiency [1].

### B. Problem Statement

Given a mathematical expression represented in infix notation and a set of input variables, the system must:

1. Parse the expression into a directed acyclic graph (DAG) representation
2. Evaluate the expression (forward pass)
3. Compute partial derivatives with respect to all input variables (backward pass)

### C. Applications

- **Neural Network Training**: Computing gradients for backpropagation algorithms
- **Optimization**: Gradient-based optimization methods (SGD, Adam, etc.)
- **Scientific Computing**: Sensitivity analysis and parameter estimation
- **Control Systems**: Computing Jacobians for system dynamics

---

## II. System Architecture

### A. Overview

The system consists of three primary components:

```
┌─────────────────────────────────────────────────────┐
│                 Input Expression                     │
│            (Infix Notation: [['exp','x'],'+',2])    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │   Builder Class      │
         │  - Parse expression  │
         │  - Build DAG         │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │  Computation Graph   │
         │  (Topological Order) │
         └─────────┬───────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
    ┌──────────┐      ┌──────────┐
    │ Forward  │      │ Backward │
    │  Pass    │      │   Pass   │
    └────┬─────┘      └─────┬────┘
         │                  │
         ▼                  ▼
    ┌─────────┐      ┌────────────┐
    │ Output  │      │ Gradients  │
    └─────────┘      └────────────┘
```

### B. Builder Class

**Functionality**: Converts infix notation to computation graph

**Algorithm**:
```
ALGORITHM: BuildGraph(infix, in_vars)
INPUT: infix expression, input variables
OUTPUT: computation graph G

1. Initialize graph G with input variables
2. FOR each element e in infix DO
3.     IF e is list THEN
4.         Recursively process e
5.         Create intermediate variable v
6.         Store operation and operands in G[v]
7.     END IF
8. END FOR
9. RETURN G
```

**Key Features**:
- Handles nested expressions recursively
- Maintains topological ordering automatically
- Supports both unary and binary operations

### C. Operator Classes

Each mathematical operation inherits from an abstract `Operator` base class and implements:

1. **Forward function** `f(a, b=None)`: Computes the operation result
2. **Derivative function** `df(a, b=None)`: Returns partial derivatives

**Implemented Operators**:

| Operator | Forward `f(a,b)` | Derivatives `df(a,b)` |
|----------|------------------|----------------------|
| **Exp**  | $e^a$ | $[e^a]$ |
| **Log**  | $\ln(a)$ | $[\frac{1}{a}]$ |
| **Sin**  | $\sin(a)$ | $[\cos(a)]$ |
| **Cos**  | $\cos(a)$ | $[-\sin(a)]$ |
| **Add**  | $a + b$ | $[1, 1]$ |
| **Sub**  | $a - b$ | $[1, -1]$ |
| **Mult** | $a \times b$ | $[b, a]$ |
| **Div**  | $\frac{a}{b}$ | $[\frac{1}{b}, -\frac{a}{b^2}]$ |
| **Pow**  | $a^b$ | $[ba^{b-1}, a^b\ln(a)]$ |

### D. Executor Class

**Forward Pass**: Evaluates nodes in topological order

```python
FOR each node v in topological_order(G):
    IF v is not input variable:
        operands = [resolve(op) for op in v.operands]
        v.value = v.operation.f(*operands)
```

**Backward Pass**: Computes gradients via reverse accumulation

```python
output_node.gradient = 1.0
FOR each node v in reverse_topological_order(G):
    IF v.operation exists:
        local_gradients = v.operation.df(*v.operand_values)
        FOR each operand op, gradient g in zip(operands, local_gradients):
            op.gradient += v.gradient * g
```

---

## III. Mathematical Foundation

### A. Chain Rule

The chain rule forms the theoretical basis for backpropagation. For a composite function $y = f(g(x))$:

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

For computation graphs with multiple paths, gradients accumulate:

$$\frac{\partial L}{\partial x_i} = \sum_{j \in \text{children}(i)} \frac{\partial L}{\partial x_j} \cdot \frac{\partial x_j}{\partial x_i}$$

### B. Reverse Mode Differentiation

Given output $y$ and inputs $\{x_1, ..., x_n\}$, reverse mode computes all partial derivatives $\frac{\partial y}{\partial x_i}$ in time proportional to a single function evaluation [2].

**Complexity Analysis**:
- Forward pass: $O(n)$ where $n$ is the number of operations
- Backward pass: $O(n)$
- Total: $O(n)$ compared to $O(n^2)$ for forward mode

---

## IV. Implementation Details

### A. Data Structures

**Computation Graph Node**:
```python
{
    'value': float,           # Computed value
    'function': str,          # Operation name
    'operands': List[str],    # Operand identifiers
    'backwards': float        # Accumulated gradient
}
```

**Variable Naming Convention**:
- Input variables: User-defined (e.g., `'x'`, `'y'`)
- Intermediate variables: Auto-generated (e.g., `'v0'`, `'v1'`)

### B. Error Handling

The system includes robust error handling for:
- Division by zero in `Div` operator
- Domain errors in `Log` operator (negative inputs)
- Special cases in `Pow` operator (negative bases)

### C. Optimization Techniques

1. **In-place Operations**: Gradients accumulated directly in graph nodes
2. **Lazy Evaluation**: Values computed only when needed
3. **Memory Efficiency**: Single graph traversal for all gradients

---

## V. Experimental Results

### A. Test Case 1: Exponential Expression

**Expression**: $f(x) = e^x + 2$  
**Input**: $x = 1.0$

**Results**:
```
Output: 4.718281828459045
Derivative ∂f/∂x: 2.718281828459045
```

**Verification**:
- Analytical: $f(1) = e^1 + 2 \approx 4.718$ ✓
- Analytical: $\frac{df}{dx} = e^x = e^1 \approx 2.718$ ✓

### B. Test Case 2: Linear Expression

**Expression**: $f(x) = 2x + 1$  
**Input**: $x = 3.0$

**Results**:
```
Output: 7.0
Derivative ∂f/∂x: 2.0
```

**Verification**:
- Analytical: $f(3) = 2(3) + 1 = 7$ ✓
- Analytical: $\frac{df}{dx} = 2$ ✓

### C. Accuracy Analysis

All test cases demonstrate machine precision accuracy ($< 10^{-15}$ relative error) compared to analytical solutions, confirming the correctness of the implementation.

---

## VI. Usage Guide

### A. Installation

```bash
# No external dependencies beyond Python standard library
python --version  # Requires Python 3.7+
```

### B. Basic Usage

```python
from nlp_assignement_1_final_handin import Builder, Executor

# Define expression: sin(x) * cos(x)
infix = [['sin', 'x'], '*', ['cos', 'x']]
in_vars = {"x": 0.5}

# Build computation graph
builder = Builder(infix, in_vars)

# Execute forward and backward passes
executor = Executor(builder.graph, in_vars)
executor.forward()
executor.backward()

print(f"f({in_vars['x']}) = {executor.output}")
print(f"df/dx = {executor.derivative['x']}")
```

### C. Supported Operations

**Unary Operations**: `exp`, `log`, `sin`, `cos`  
**Binary Operations**: `+`, `-`, `*`, `/`, `^`

**Expression Format**:
- Unary: `['operation', operand]`
- Binary: `[operand1, 'operation', operand2]`
- Nesting: Use lists within lists for sub-expressions

---

## VII. Limitations and Future Work

### A. Current Limitations

1. **Single Output**: System supports only scalar-valued functions
2. **Static Graphs**: Graph must be rebuilt for different expressions
3. **No Optimization**: No constant folding or expression simplification

### B. Proposed Extensions

1. **Multi-Output Support**: Implement Jacobian computation for vector-valued functions
2. **Dynamic Graphs**: Support for control flow (if statements, loops)
3. **Higher-Order Derivatives**: Extend to compute Hessians
4. **GPU Acceleration**: Port to frameworks like PyTorch or JAX
5. **Expression Optimization**: Implement common subexpression elimination

---

## VIII. Conclusion

This project successfully implements a complete automatic differentiation system using reverse-mode differentiation. The system accurately computes both function values and gradients for arbitrary mathematical expressions, demonstrating the fundamental concepts underlying modern deep learning frameworks. The modular architecture allows easy extension to additional operations and optimization techniques.

The implementation validates the efficiency of reverse-mode differentiation for computing gradients, requiring only $O(n)$ time regardless of the number of input variables—a crucial property for scaling to high-dimensional optimization problems in machine learning.

---

## IX. References

[1] A. Griewank and A. Walther, "Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation," SIAM, 2nd ed., 2008.

[2] A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind, "Automatic differentiation in machine learning: a survey," *Journal of Machine Learning Research*, vol. 18, no. 153, pp. 1-43, 2018.

[3] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436-444, 2015.

[4] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, no. 6088, pp. 533-536, 1986.

---

## X. Appendix

### A. File Structure

```
Assignment-1/
├── README.md                              # This document
├── nlp_assignement_1_final_handin.ipynb  # Main implementation
├── backprop.py                           # Standalone Python module
├── Assignment 1.pdf                       # Assignment specification
└── NLP_CA1_report.pdf                    # Project report
```

### B. Code Repository

**Repository**: [Natural-Language-Processing-Projects-SUT](https://github.com/tahamajs/Natural-Language-Processing-Projects-SUT)  
**Branch**: main  
**Path**: `/nlp-projects/Assignment-1/`

### C. Contact Information

**Author**: Taha Majlesi  
**Email**: [Your Email]  
**Institution**: Sharif University of Technology  
**Course**: Natural Language Processing

---

## License

This project is part of academic coursework and is provided for educational purposes under the MIT License.

---

*Last Updated: October 2, 2025*  
*Document Version: 1.0*
