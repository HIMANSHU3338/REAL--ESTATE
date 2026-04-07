import gradio as gr

def predict(area, price_per_sqft):
    total = area * price_per_sqft
    return f"Estimated Property Value: ₹{total}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Area (sqft)"),
        gr.Number(label="Price per sqft (₹)")
    ],
    outputs="text",
    title="Real Estate Price Calculator"
)

demo.launch()