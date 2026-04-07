import gradio as gr

def predict(area, price):
    return f"Estimated Property Price: ₹{area * price}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Area (sqft)"),
        gr.Number(label="Price per sqft (₹)")
    ],
    outputs="text",
    title="Real Estate Calculator"
)

demo.launch()