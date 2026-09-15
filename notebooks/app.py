import streamlit as st
from model import predict

st.set_page_config(page_title="Industrial Predictive Maintenance", page_icon="🏭")
st.title("🏭 Industrial Predictive Maintenance")
st.write("Enter machine sensor readings to estimate failure risk.")

c1, c2 = st.columns(2)
with c1:
    type_ = st.selectbox("Type", ["L", "M", "H"])
    air_temp = st.number_input("Air temperature [K]", value=300.0)
    process_temp = st.number_input("Process temperature [K]", value=310.0)
with c2:
    rotational_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
    torque = st.number_input("Torque [Nm]", value=40.0)
    tool_wear = st.number_input("Tool wear [min]", value=100.0)

if st.button("Predict"):
    result = predict({
        "Type": type_,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    })
    st.metric("Failure probability", f"{result['failure_probability']:.2%}")
    st.write("Optimized threshold:", f"{result['threshold']:.3f}")
    if result["failure_prediction"]:
        st.error("⚠️ Failure Risk")
    else:
        st.success("✅ Normal")
