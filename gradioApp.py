from app import classify_text_api,TextData
import gradio as gr
import pandas as pd
# to run open terminal and write: python gradioApp.py
#close server ctrl+c
# Function to generate bar plot based on user selection
def bar_plot_fn(text, kind,threshold=0.65):
    # Define functions for generating bar plots
    def above_threshold_bar_plot():
        classes = classify_text_api(TextData(text=text,threshold=threshold))
        keys = list(classes.keys())
        vals = list(classes.values())
        res = pd.DataFrame({"classes": keys, "values": vals})
        above_threshold = res
        plot = gr.BarPlot(
            above_threshold,
            x="classes",
            y="values",
            width=1000,
            title="Classes Above Threshold",
            tooltip=["classes", "values"],
            y_lim=[0, 1]
        )
        return plot

    def below_threshold_bar_plot():
        classes = classify_text_api(TextData(text=text, threshold=threshold))
        keys = list(classes.keys())
        vals = list(classes.values())
        res = pd.DataFrame({"classes": keys, "values": vals})
        below_threshold = res
        plot = gr.BarPlot(
            below_threshold,
            x="classes",
            y="values",
            width=1000,
            title="Classes Below Threshold",
            tooltip=["classes", "values"],
            y_lim=[0, 1]
        )
        return plot

    def equal_to_zero_bar_plot():
        classes = classify_text_api(TextData(text=text,threshold=threshold))
        keys = list(classes.keys())
        vals = list(classes.values())
        res = pd.DataFrame({"classes": keys, "values": vals})
        equal_to_zero = res
        plot = gr.BarPlot(
            equal_to_zero,
            x="classes",
            y="values",
            width=1000,
            title="Classes Equal to Zero",
            tooltip=["classes", "values"],
            y_lim=[0, 1]
        )
        return plot

    # Determine which function to call based on user input
    if text:
        print(text)
        if kind == "above":
            return above_threshold_bar_plot()
        if kind == "below":
            return below_threshold_bar_plot()
        else:
            return equal_to_zero_bar_plot()

    else:
        return gr.Row([])



iface = gr.Interface(
    fn=bar_plot_fn,
    inputs=[gr.Textbox(label="Enter your message", placeholder="Text to analyze"), 
            gr.Dropdown( ["above", "below","zero"], value=["above"]),
            #gr.Slider(0, 1,step=0.1, label="Count", info="Choose between 0 and 1")
            ],
    outputs="barplot",
    title="Text Classification Dashboard",
    description="This dashboard classifies text into different categories based on confidence scores.",
    allow_flagging="never"
)

# # Launch the interface
iface.launch(share=True)



