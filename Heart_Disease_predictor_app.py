import gradio as gr
from heart_disease_predictor import HeartDiseasePredictor

# Load predictor
predictor = HeartDiseasePredictor()

# Get input fields and their descriptions
feature_info = predictor.get_feature_info()

# Create inputs dynamically
input_components = []
for key, info in feature_info.items():
    if info["type"] == "int":
        input_components.append(gr.Slider(minimum=info["min"], maximum=info["max"], step=1, label=f"{key} - {info['description']}"))
    elif info["type"] == "float":
        input_components.append(gr.Slider(minimum=info["min"], maximum=info["max"], step=0.1, label=f"{key} - {info['description']}"))
    elif info["type"] == "binary":
        input_components.append(gr.Radio(choices=info["values"], label=f"{key} - {info['description']}"))
    elif info["type"] == "categorical":
        input_components.append(gr.Dropdown(choices=info["values"], label=f"{key} - {info['description']}"))

# Prediction function for Gradio
def predict_heart_disease(*inputs):
    try:
        # Map inputs to dictionary
        input_data = dict(zip(predictor.feature_names, inputs))
        result = predictor.predict(input_data)

        if "error" in result:
            return f"❌ Error: {result['error']}", "", "", ""

        return (
            f"{result['probability']:.2f}",
            result["risk_level"],
            result["recommendation"],
            result["final_tip"]
        )

    except Exception as e:
        return f"❌ Exception occurred: {e}", "", "", ""

# Interface
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=input_components,
    outputs=[
        gr.Textbox(label="🩺 Probability of Heart Disease"),
        gr.Textbox(label="🔍 Risk Level"),
        gr.Textbox(label="📌 Recommendation"),
        gr.Textbox(label="💡Health Tip"),
    ],
    title="💖 Heart Disease Risk Predictor",
    description="Enter the patient's medical details below to estimate the risk of heart disease. Please consult a doctor for final diagnosis.",
    theme="soft",
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
