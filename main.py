import os
from dotenv import load_dotenv
import gradio as gr
import openai
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

class OpenAITokenPredictionBrainMap:
    def __init__(self):
        """Initialize with OpenAI API key from .env file."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file or environment variables")

        self.client = openai.OpenAI(api_key=api_key)
        print(f"✅ Successfully loaded API key from .env file")

    def get_token_predictions_with_logprobs(self, prompt: str, max_tokens: int = 5,
                                            top_logprobs: int = 3, model: str = "gpt-3.5-turbo") -> tuple:
        """Get token predictions with log probabilities from OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=top_logprobs,
                temperature=0.7,
                stream=False
            )

            generated_text = response.choices[0].message.content
            logprobs_data = response.choices[0].logprobs

            if not logprobs_data or not logprobs_data.content:
                return [], generated_text

            prediction_tree = []

            for step_idx, token_data in enumerate(logprobs_data.content):
                token = token_data.token
                logprob = token_data.logprob
                probability = np.exp(logprob)

                top_alternatives = []
                if token_data.top_logprobs:
                    for alt_token_data in token_data.top_logprobs:
                        alt_token = alt_token_data.token
                        alt_logprob = alt_token_data.logprob
                        alt_probability = np.exp(alt_logprob)
                        top_alternatives.append({
                            'token': alt_token,
                            'probability': float(alt_probability),
                            'logprob': float(alt_logprob),
                            'is_selected': alt_token == token
                        })

                top_alternatives.sort(key=lambda x: x['probability'], reverse=True)

                step_data = {
                    'step': step_idx + 1,
                    'selected_token': token,
                    'selected_probability': float(probability),
                    'selected_logprob': float(logprob),
                    'predictions': top_alternatives,
                    'context': prompt + " " + "".join([t.token for t in logprobs_data.content[:step_idx + 1]])
                }
                prediction_tree.append(step_data)

            return prediction_tree, generated_text

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return [], f"Error: {str(e)}"

    def create_scrollable_brainmap(self, prediction_tree: List[Dict]) -> go.Figure:
        """Create fully interactive scrollable brainmap visualization."""
        if not prediction_tree:
            fig = go.Figure()
            fig.add_annotation(
                text="No prediction data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="red")
            )
            return fig

        fig = go.Figure()

        # Enhanced color scheme
        colors = {
            'selected': '#FF6B6B',
            'alternative': '#4ECDC4',
            'edge_selected': '#FF4444',
            'edge_alternative': '#45B7AA',
            'background': '#f8f9fa'
        }

        # Calculate layout with more spacing for scrolling
        horizontal_spacing = 4  # Increased spacing between steps
        vertical_spacing = 1.2  # Increased spacing between alternatives

        node_positions = {}

        # Add nodes with enhanced positioning
        for step_idx, step_data in enumerate(prediction_tree):
            step_num = step_data['step']
            x_pos = step_idx * horizontal_spacing

            for pred_idx, prediction in enumerate(step_data['predictions']):
                token = prediction['token']
                prob = prediction['probability']
                is_selected = prediction['is_selected']

                # Enhanced vertical positioning
                if is_selected:
                    y_pos = 3  # Selected token at top
                else:
                    y_pos = 3 - (pred_idx * vertical_spacing)

                node_id = f"step_{step_num}_token_{token}"
                node_positions[node_id] = (x_pos, y_pos)

                # Enhanced node styling
                color = colors['selected'] if is_selected else colors['alternative']
                size = 30 + prob * 60  # Larger nodes for better visibility

                # Add enhanced hover information
                hover_text = (
                    f"<b>Step {step_num}</b><br>"
                    f"Token: '<b>{token}</b>'<br>"
                    f"Probability: <b>{prob:.4f}</b><br>"
                    f"Log Probability: {prediction.get('logprob', 'N/A'):.3f}<br>"
                    f"Rank: {pred_idx + 1}<br>"
                    f"Selected: {'✅ Yes' if is_selected else '❌ No'}"
                )

                fig.add_trace(go.Scatter(
                    x=[x_pos], y=[y_pos],
                    mode='markers+text',
                    marker=dict(
                        size=size,
                        color=color,
                        line=dict(
                            width=4 if is_selected else 2,
                            color='black' if is_selected else 'gray'
                        ),
                        opacity=0.95 if is_selected else 0.75
                    ),
                    text=f"<b>{token}</b><br>{prob:.3f}" if is_selected else f"{token}<br>{prob:.3f}",
                    textposition="middle center",
                    textfont=dict(
                        size=11 if is_selected else 9,
                        color='white' if is_selected else 'black',
                        family="Arial Black" if is_selected else "Arial"
                    ),
                    name=f"Step {step_num}: {token}",
                    hovertemplate=hover_text + "<extra></extra>",
                    showlegend=False
                ))

        # Add enhanced edges
        for step_idx in range(len(prediction_tree) - 1):
            current_step = prediction_tree[step_idx]
            next_step = prediction_tree[step_idx + 1]

            selected_token = current_step['selected_token']
            current_node_id = f"step_{current_step['step']}_token_{selected_token}"

            if current_node_id in node_positions:
                current_pos = node_positions[current_node_id]

                for next_prediction in next_step['predictions']:
                    next_token = next_prediction['token']
                    next_prob = next_prediction['probability']
                    is_selected_edge = next_prediction['is_selected']

                    next_node_id = f"step_{next_step['step']}_token_{next_token}"
                    if next_node_id in node_positions:
                        next_pos = node_positions[next_node_id]

                        # Enhanced edge styling
                        edge_color = colors['edge_selected'] if is_selected_edge else colors['edge_alternative']
                        edge_width = 8 if is_selected_edge else 3
                        opacity = 0.9 if is_selected_edge else 0.5

                        edge_hover = (
                            f"<b>Transition Path</b><br>"
                            f"From: '<b>{selected_token}</b>'<br>"
                            f"To: '<b>{next_token}</b>'<br>"
                            f"Probability: <b>{next_prob:.4f}</b><br>"
                            f"Path Type: {'🎯 Selected' if is_selected_edge else '💭 Alternative'}"
                        )

                        fig.add_trace(go.Scatter(
                            x=[current_pos[0], next_pos[0]],
                            y=[current_pos[1], next_pos[1]],
                            mode='lines',
                            line=dict(width=edge_width, color=edge_color),
                            opacity=opacity,
                            hovertemplate=edge_hover + "<extra></extra>",
                            showlegend=False
                        ))

        # CORRECTED: Enhanced layout with proper Plotly properties only
        fig.update_layout(
            title={
                'text': "🧠 Interactive Token Prediction Brainmap (Scroll, Zoom, Pan Enabled)",
                'x': 0.5,
                'font': {'size': 20, 'family': 'Arial Black', 'color': '#2C3E50'}
            },
            xaxis=dict(
                title="Generation Steps →",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                tickmode='array',
                tickvals=list(range(0, len(prediction_tree) * horizontal_spacing, horizontal_spacing)),
                ticktext=[f"Step {i + 1}" for i in range(len(prediction_tree))],
                range=[-1, (len(prediction_tree) - 1) * horizontal_spacing + 1],
                fixedrange=False  # Allow zooming and panning on X-axis
            ),
            yaxis=dict(
                title="Token Alternatives ↑ (Selected → Alternatives)",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                range=[-3, 4],
                fixedrange=False  # Allow zooming and panning on Y-axis
            ),
            plot_bgcolor='white',
            paper_bgcolor=colors['background'],
            # Enhanced size for better scrolling experience
            width=1200,  # Wider for horizontal scrolling
            height=700,  # Taller for vertical scrolling
            hovermode='closest',
            margin=dict(l=80, r=80, t=100, b=80),

            # CORRECTED: Valid layout properties only
            dragmode='pan',  # Default to pan mode
            # REMOVED: scrollZoom=True,  # This is NOT a valid layout property
            # REMOVED: doubleClick='reset',  # This is NOT a valid layout property
            # REMOVED: showTips=True,  # This is NOT a valid layout property

            # Add range slider for easy navigation
            xaxis_rangeslider=dict(
                visible=True,
                thickness=0.05
            ),

            # Add modebar with all tools
            modebar=dict(
                orientation='h',
                bgcolor='rgba(255,255,255,0.8)',
                color='#444',
                activecolor='#FF6B6B'
            )
        )

        # CORRECTED: Scroll zoom is controlled by Gradio Plot component, not layout
        fig.update_traces(
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=12,
                font_family="Arial"
            )
        )

        return fig


def create_gradio_interface():
    """Create the main Gradio interface with enhanced scrollable visualization."""

    api_key_status = "✅ Connected" if os.getenv('OPENAI_API_KEY') else "❌ Not Found"

    try:
        brainmap = OpenAITokenPredictionBrainMap()
        initialization_status = "✅ OpenAI API Client Initialized Successfully"
    except Exception as e:
        brainmap = None
        initialization_status = f"❌ Error: {str(e)}"

    def process_prediction(prompt: str, model_choice: str, max_tokens: int, top_logprobs: int):
        """Process the prediction and return output text and scrollable visualization."""
        if brainmap is None:
            return "❌ OpenAI API client not initialized.", None, "API client initialization failed."

        if not prompt.strip():
            return "❌ Please enter a prompt", None, "Please provide a prompt."

        try:
            prediction_tree, generated_text = brainmap.get_token_predictions_with_logprobs(
                prompt=prompt.strip(),
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
                model=model_choice
            )

            if not prediction_tree:
                return f"Generated Text: {generated_text}", None, "No prediction data available."

            # Create scrollable brainmap
            fig = brainmap.create_scrollable_brainmap(prediction_tree)

            # Format detailed output
            output_text = f"✅ **Generated Text:**\n{generated_text}\n\n"
            output_text += f"📊 **Analysis Summary:**\n"
            output_text += f"• Total Steps: {len(prediction_tree)}\n"
            output_text += f"• Model: {model_choice}\n\n"

            output_text += f"🔍 **Step-by-Step Analysis:**\n"
            for step_data in prediction_tree:
                selected = step_data['selected_token']
                prob = step_data['selected_probability']
                alternatives = [p for p in step_data['predictions'] if not p['is_selected']][:2]

                output_text += f"**Step {step_data['step']}:** '{selected}' ({prob:.3f})"
                if alternatives:
                    alt_text = ", ".join([f"'{alt['token']}' ({alt['probability']:.3f})" for alt in alternatives])
                    output_text += f"\n   📋 Alternatives: {alt_text}"
                output_text += "\n"

            avg_confidence = np.mean([step['selected_probability'] for step in prediction_tree])
            output_text += f"\n📈 **Average Confidence:** {avg_confidence:.3f}"

            # Interactive instructions
            instructions = (
                "🎯 **How to Navigate the Brainmap:**\n"
                "• **Scroll Wheel:** Zoom in/out\n"
                "• **Click & Drag:** Pan around the diagram\n"
                "• **Double-click:** Reset zoom to fit\n"
                "• **Range Slider:** Navigate horizontally (bottom)\n"
                "• **Hover:** Get detailed token information\n"
                "• **Toolbar:** Use zoom, pan, select tools (top-right)"
            )

            return output_text + f"\n\n{instructions}", fig, f"✅ Interactive brainmap ready! {len(prediction_tree)} steps generated."

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return error_msg, None, f"Error: {str(e)}"

    # Enhanced CSS for scrollable interface
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        max-width: 100% !important;
    }
    .input-container {
        background-color: #f0f2f6 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    .output-container {
        background-color: #e8f5e8 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    .plot-container {
        border: 2px solid #ddd !important;
        border-radius: 10px !important;
        overflow: auto !important;
        background-color: white !important;
    }
    """

    with gr.Blocks(
            title="🧠 OpenAI Token Prediction Brainmap - Scrollable",
            theme=gr.themes.Soft(),
            css=custom_css,
            fill_width=True  # Use full browser width
    ) as demo:

        gr.Markdown(
            f"""
            # 🧠 OpenAI Token Prediction Brainmap Visualizer (Interactive & Scrollable)

            **Fully interactive brainmap with scroll, zoom, and pan capabilities**

            🔑 **API Status:** {api_key_status} | **Client Status:** {initialization_status}

            🎯 **Navigation:** Scroll to zoom, drag to pan, double-click to reset, use range slider to navigate!
            """
        )

        # Upper half: Input and Output
        with gr.Row():
            with gr.Column(scale=1, elem_classes="input-container"):
                gr.Markdown("### 📝 Input Configuration")

                prompt_input = gr.Textbox(
                    label="💭 Your Prompt",
                    placeholder="Enter your text prompt here...",
                    lines=4,
                    value="The future of artificial intelligence will revolutionize",
                    info="Text prompt to analyze for token predictions"
                )

                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"],
                        value="gpt-3.5-turbo",
                        label="🤖 Model"
                    )

                    max_tokens_slider = gr.Slider(
                        minimum=3, maximum=100, value=25, step=1,
                        label="📏 Max Tokens"
                    )

                top_logprobs_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="🔝 Top Alternatives"
                )

                analyze_btn = gr.Button(
                    "🚀 Generate Interactive Brainmap",
                    variant="primary", size="lg"
                )

            with gr.Column(scale=1, elem_classes="output-container"):
                gr.Markdown("### 📊 Analysis Results")

                output_text = gr.Textbox(
                    label="Generated Text & Navigation Guide",
                    lines=20, max_lines=30,
                    interactive=False,
                    show_copy_button=True
                )

                status_text = gr.Textbox(
                    label="Status", lines=1, interactive=False
                )

        # Lower half: Enhanced scrollable visualization
        gr.Markdown("### 🧠 Interactive Token Prediction Brainmap")
        gr.Markdown(
            "*🎯 **Fully Interactive:** Scroll wheel to zoom, click & drag to pan, hover for details, "
            "use toolbar for advanced controls, range slider for navigation*"
        )

        # CORRECTED: Gradio Plot component automatically handles scrolling
        prediction_plot = gr.Plot(
            label="Scrollable Token Prediction Brainmap",
            visible=True,
            elem_classes="plot-container"
        )

        # Event handlers
        analyze_btn.click(
            fn=process_prediction,
            inputs=[prompt_input, model_choice, max_tokens_slider, top_logprobs_slider],
            outputs=[output_text, prediction_plot, status_text]
        )

        # Auto-load with demo
        demo.load(
            fn=process_prediction,
            inputs=[prompt_input, model_choice, max_tokens_slider, top_logprobs_slider],
            outputs=[output_text, prediction_plot, status_text]
        )

        # Enhanced examples section
        with gr.Accordion("📚 Interactive Features & Examples", open=False):
            gr.Markdown(
                """
                ### 🎮 **Interactive Controls:**
                - **🖱️ Scroll Wheel:** Zoom in/out on any part of the diagram
                - **👆 Click & Drag:** Pan around to explore different areas  
                - **🖱️ Double-click:** Reset zoom to show entire diagram
                - **📏 Range Slider:** Navigate horizontally (at bottom of plot)
                - **🔧 Toolbar:** Access zoom, pan, select, and download tools
                - **ℹ️ Hover:** Get detailed information about tokens and transitions

                ### 🎯 **Example Prompts:**
                - **Long Sequence:** "Once upon a time in a magical forest where ancient trees whispered secrets"
                - **Technical:** "Machine learning algorithms optimize neural networks through gradient descent and backpropagation"
                - **Creative:** "The mysterious door appeared suddenly in the middle of the empty field"
                - **Conversational:** "What are the most important factors to consider when choosing a career path"

                ### 💡 **Pro Tips:**
                - **More Tokens (8-20):** Creates larger, more scrollable diagrams
                - **More Alternatives (4-5):** Shows richer decision trees
                - **Different Models:** Compare how different GPT models make predictions
                - **Save Plot:** Use toolbar download button to save your brainmap
                """
            )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )
