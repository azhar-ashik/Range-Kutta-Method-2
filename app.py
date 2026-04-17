import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import sympify, symbols, lambdify

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Numerical Methods | RK4 Solver",
    page_icon="🔬",
    layout="wide"
)

# --- MODERN STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR & INSTRUCTIONS ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    with st.expander("📝 Input Instructions", expanded=True):
        st.info("""
        **How to enter equations:**
        - **Power:** `x^2` or `x**2`
        - **Product:** `2*x*y`
        - **Functions:** `sin(x)`, `exp(x)`, `sqrt(x)`
        - **Variables:** Use only `x` and `y`.
        """)

    eq_input = st.text_input("Enter dy/dx = f(x, y)", "x + y^2")
    
    col_a, col_b = st.columns(2)
    with col_a:
        x0 = st.number_input("x₀ (Initial)", value=0.0)
        y0 = st.number_input("y₀ (Initial)", value=1.0)
    with col_b:
        xn = st.number_input("xₙ (Target)", value=1.0)
        h = st.number_input("Step Size (h)", value=0.1, format="%.3f")

    solve_btn = st.button("🚀 Run Simulation")

# --- RK4 ENGINE ---
def run_rk4(equation_str, x0, y0, xn, h):
    # Standardize notation (x^2 -> x**2)
    cleaned_eq = equation_str.replace('^', '**')
    x, y = symbols('x y')
    
    try:
        expr = sympify(cleaned_eq)
        # Check for invalid variables
        invalid_vars = [str(v) for v in expr.free_symbols if str(v) not in ['x', 'y']]
        if invalid_vars:
            return None, f"Invalid variables: {', '.join(invalid_vars)}"
            
        f = lambdify((x, y), expr, "numpy")
    except Exception as e:
        return None, f"Equation Error: {e}"

    x_vals, y_vals = [x0], [y0]
    curr_x, curr_y = x0, y0
    steps = int(abs(xn - x0) / h)

    for _ in range(steps):
        k1 = f(curr_x, curr_y)
        k2 = f(curr_x + 0.5*h, curr_y + 0.5*h*k1)
        k3 = f(curr_x + 0.5*h, curr_y + 0.5*h*k2)
        k4 = f(curr_x + h, curr_y + h*k3)
        
        curr_y = curr_y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        curr_x = curr_x + h
        
        x_vals.append(curr_x)
        y_vals.append(curr_y)
        
    return pd.DataFrame({"Step": range(len(x_vals)), "x": x_vals, "y": y_vals}), None

# --- MAIN UI LOGIC ---
st.title("🔢 Fourth-Order Runge-Kutta Method")
st.caption("Numerical Solution of Ordinary Differential Equations")

if solve_btn:
    with st.spinner('Calculating...'):
        df, error = run_rk4(eq_input, x0, y0, xn, h)
        
        if error:
            st.error(error)
        else:
            # Metrics display
            m1, m2, m3 = st.columns(3)
            m1.metric("Start Point", f"({x0}, {y0})")
            m2.metric("Final Result (y)", f"{df['y'].iloc[-1]:.6f}")
            m3.metric("Total Iterations", len(df)-1)

            st.markdown("---")
            
            # Layout for Table and Graph
            tab_col, plot_col = st.columns([1, 2])
            
            with tab_col:
                st.subheader("📊 Iteration Table")
                st.dataframe(df.set_index('Step'), use_container_width=True)
                
            with plot_col:
                st.subheader("📈 Visual Solution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['x'], df['y'], color='#1E88E5', linewidth=2, label='RK4 Path')
                ax.scatter(df['x'], df['y'], color='#D81B60', s=25)
                ax.set_title(f"Numerical Integration for f(x,y) = {eq_input}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
else:
    st.write("👈 Configure the parameters in the sidebar and click 'Run Simulation' to begin.")
