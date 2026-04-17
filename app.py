import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import sympify, symbols, lambdify

# Page Config for a professional feel
st.set_page_config(
    page_title="RK4 Numerical Solver",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Runge-Kutta 4th Order Implementation")
st.markdown("---")

# Sidebar - Modernized with better grouping
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/mathematics.png", width=80)
    st.header("Configuration")
    
    with st.expander("Equation Settings", expanded=True):
        eq_input = st.text_input("dy/dx = f(x, y)", "x + y", help="Use standard math notation like x**2, sin(x), etc.")
        h = st.number_input("Step Size (h)", value=0.1, step=0.01, format="%.3f")
    
    with st.expander("Initial & Boundary Conditions", expanded=True):
        x0 = st.number_input("Initial x₀", value=0.0)
        y0 = st.number_input("Initial y₀", value=1.0)
        xn = st.number_input("Target xₙ", value=2.0)

    solve_btn = st.button("Calculate Solution")

def rk4_engine(equation_str, x0, y0, xn, h):
    x, y = symbols('x y')
    try:
        expr = sympify(equation_str)
        f = lambdify((x, y), expr, "numpy")
    except Exception as e:
        return None, f"Expression Error: {e}"

    x_vals, y_vals = [x0], [y0]
    curr_x, curr_y = x0, y0
    steps = int((xn - x0) / h)

    for _ in range(steps):
        k1 = f(curr_x, curr_y)
        k2 = f(curr_x + 0.5*h, curr_y + 0.5*h*k1)
        k3 = f(curr_x + 0.5*h, curr_y + 0.5*h*k2)
        k4 = f(curr_x + h, curr_y + h*k3)
        
        curr_y = curr_y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        curr_x = curr_x + h
        
        x_vals.append(curr_x)
        y_vals.append(curr_y)
        
    return pd.DataFrame({"x": x_vals, "y": y_vals}), None

if solve_btn:
    data, error = rk4_engine(eq_input, x0, y0, xn, h)
    
    if error:
        st.error(error)
    else:
        # Modern Metric Display
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Final x", f"{data['x'].iloc[-1]:.2f}")
        col_m2.metric("Final y (Result)", f"{data['y'].iloc[-1]:.4f}")
        col_m3.metric("Steps Taken", len(data)-1)

        st.markdown("### Visual Analysis")
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.write("#### Data Table")
            st.dataframe(data.style.highlight_max(axis=0), use_container_width=True, height=400)
            
        with col_right:
            # Modern Plotting with Matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data['x'], data['y'], color='#007bff', linewidth=2, label='RK4 Approximation')
            ax.scatter(data['x'], data['y'], color='#ff7f0e', s=30)
            
            ax.set_title(f"Solution for dy/dx = {eq_input}", fontsize=14, fontweight='bold')
            ax.set_xlabel("x (Independent Variable)")
            ax.set_ylabel("y (Solution)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
            st.pyplot(fig)

        # Download option for project reports
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv, file_name="rk4_results.csv", mime="text/csv")
