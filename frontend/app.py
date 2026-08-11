import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="SwiftDesk IT Support Assistant",
    page_icon="🛠️",
    layout="wide",
)


st.title("SwiftDesk IT Support Assistant")
st.write(
    "Draft IT support responses using historical support tickets. "
    "All generated responses require human review before being sent."
)


st.subheader("Customer Issue")

customer_issue = st.text_area(
    "Describe the customer's issue:",
    height=150,
    placeholder="Example: My computer cannot connect to Wi-Fi.",
)


col1, col2 = st.columns(2)

with col1:
    prompt_style = st.selectbox(
        "Prompt style",
        options=[
            "zero-shot",
            "few-shot",
            "reasoned",
            "rag",
        ],
    )

with col2:
    rag_enabled = st.checkbox(
        "Enable RAG",
        value=True,
    )


num_examples = st.slider(
    "Number of retrieved examples",
    min_value=1,
    max_value=5,
    value=3,
)


if st.button("Generate Support Draft", type="primary"):
    if not customer_issue.strip():
        st.warning("Please enter a customer issue.")
    else:
        request_data = {
            "customer_issue": customer_issue,
            "prompt_style": prompt_style,
            "rag_enabled": rag_enabled,
            "num_examples": num_examples,
        }

        try:
            response = requests.post(
                f"{API_URL}/generate",
                json=request_data,
                timeout=60,
            )

            if response.status_code != 200:
                st.error(
                    f"API error: {response.status_code}\n\n"
                    f"{response.text}"
                )
            else:
                result = response.json()

                st.subheader("Draft Support Reply")

                st.info(result["draft_reply"])

                sources = result.get("retrieved_sources", [])

                if rag_enabled:
                    st.subheader("Retrieved Sources")

                    if not sources:
                        st.write("No sources were retrieved.")
                    else:
                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):
                            with st.expander(
                                f"Source {index}"
                            ):
                                st.markdown(
                                    "**Previous customer issue:**"
                                )
                                st.write(
                                    source["customer_issue"]
                                )

                                st.markdown(
                                    "**Approved support reply:**"
                                )
                                st.write(
                                    source["reference_reply"]
                                )

                st.warning(
                    "Human review is required before sending "
                    "this response to a customer."
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend. "
                "Make sure the backend is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The request timed out. "
                "Please try again."
            )

