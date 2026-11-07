
import streamlit as st


def get_chatbot_answer(question, current_assessment=None):
    question = question.lower().strip()

    # 1. Emergency guidance
    emergency_words = [
        "180", "emergency", "chest pain",
        "difficulty breathing", "cannot breathe"
    ]

    if any(word in question for word in emergency_words):
        return (
            "A blood pressure reading above 180 systolic and/or "
            "120 diastolic needs urgent attention. If possible, "
            "repeat the measurement after at least one minute. "
            "If it remains very high, contact a healthcare "
            "professional immediately. If you have chest pain, "
            "difficulty breathing, weakness, confusion or vision "
            "changes, seek emergency medical help immediately."
        )

    # 2. Basic hypertension questions
    if "hypertension" in question and any(
        word in question for word in ["what is", "meaning", "mean"]
    ):
        return (
            "Hypertension means persistently high blood pressure. "
            "It is the force of blood pushing against artery walls. "
            "If blood pressure remains too high, it can increase "
            "the risk of heart disease, stroke and kidney damage."
        )

    if "symptom" in question or "signs of hypertension" in question:
        return (
            "Hypertension often has no noticeable symptoms. "
            "Measuring blood pressure is important, and a healthcare "
            "professional can help confirm whether it is persistently high."
        )

    if "cause" in question or "risk factor" in question:
        return (
            "Risk factors can include high salt intake, smoking, "
            "low physical activity, family history, increasing age, "
            "some medical conditions and other lifestyle factors. "
            "Risk factors do not prove that someone has hypertension."
        )

    if "salt" in question:
        return (
            "Too much salt can contribute to higher blood pressure. "
            "Reducing salt in your diet may help support healthy "
            "blood pressure."
        )

    if "smoke" in question:
        return (
            "Smoking can raise blood pressure temporarily and damages "
            "blood vessels. Avoiding tobacco supports cardiovascular health."
        )

    if "stress" in question:
        return (
            "Stress can temporarily raise blood pressure. Long-term "
            "stress may also affect sleep and lifestyle habits. "
            "Relaxation, regular activity and professional support "
            "may help manage stress."
        )

    if "exercise" in question or "physical activity" in question:
        return (
            "Regular physical activity can help support healthy blood "
            "pressure and heart health. Choose activities suitable for "
            "your abilities and ask a healthcare professional if unsure."
        )

    if "sleep" in question:
        return (
            "Poor sleep can be associated with high blood pressure. "
            "A consistent sleep routine and adequate sleep can support "
            "general health."
        )

    if "family history" in question or "hereditary" in question:
        return (
            "Having close family members with hypertension can increase "
            "your risk, but it does not mean you will definitely develop it."
        )

    # 3. Understanding blood pressure
    if "systolic" in question or "diastolic" in question:
        return (
            "Systolic pressure is the top number and measures pressure "
            "when the heart contracts. Diastolic pressure is the bottom "
            "number and measures pressure when the heart relaxes."
        )

    if "mmhg" in question:
        return (
            "mmHg means millimetres of mercury. It is the unit commonly "
            "used to report blood pressure."
        )

    if "normal blood pressure" in question or "healthy blood pressure" in question:
        return (
            "For many adults, a reading below 120/80 mmHg is considered "
            "normal by common guidelines. Interpretation depends on the "
            "guideline and individual circumstances. One reading alone "
            "does not establish a diagnosis."
        )

    if "measure" in question or "monitor" in question:
        return (
            "For a more reliable measurement, rest quietly for five "
            "minutes, sit with your back supported and feet flat, keep "
            "your arm supported at heart level, and use a correctly "
            "sized cuff. Follow the monitor's instructions."
        )

    if "hypotension" in question:
        return (
            "Hypotension means low blood pressure. Whether it causes "
            "a problem depends on the person and symptoms. Seek medical "
            "advice for concerning symptoms such as fainting or dizziness."
        )

    # 4. Lifestyle and treatment
    if "prevent" in question or "lower blood pressure" in question:
        return (
            "Helpful measures may include reducing salt, eating a "
            "balanced diet, exercising regularly, avoiding smoking and "
            "following professional medical advice. Never stop or change "
            "prescribed medication without consulting your healthcare "
            "professional."
        )

    if "medication" in question or "medicine" in question or "tablets" in question:
        return (
            "Blood pressure medicines can help control hypertension. "
            "The appropriate medicine depends on the individual. Take "
            "prescribed medicine as directed and consult a healthcare "
            "professional before making changes."
        )

    if "kidney" in question or "stroke" in question or "heart disease" in question:
        return (
            "Persistently high blood pressure can damage blood vessels "
            "and increase the risk of stroke, heart disease and kidney "
            "problems. Monitoring and appropriate management can reduce "
            "long-term risks."
        )

    # 5. Your VUT Pharmacy project
    if "synthetic data" in question or "artificial data" in question:
        return (
            "Synthetic data is artificially generated information rather "
            "than measurements collected directly from real patients. "
            "This project uses synthetic demonstration data, so its "
            "performance does not establish clinical accuracy."
        )

    if "decision tree" in question:
        return (
            "A Decision Tree is a machine-learning model that uses "
            "learned decision rules to predict a category from input "
            "features. In this project, it produces an experimental "
            "hypertension demonstration category."
        )

    if "machine learning" in question or "how does the model work" in question:
        return (
            "Machine learning uses data to train a computer model to "
            "recognise patterns and make predictions. This project's "
            "model is educational and has not been clinically validated."
        )

    if "clinically validated" in question or "accurate is the model" in question:
        return (
            "The model has not been clinically validated. It uses "
            "synthetic data and should not be used to diagnose, treat "
            "or make real patient-care decisions."
        )

    if "risk score" in question or "risk factor score" in question:
        return (
            "The demonstration risk score counts selected risk factors "
            "included in the project's scoring rules. It is an educational "
            "score, not a validated clinical risk calculator."
        )

    if "my result" in question or "my assessment" in question or "my prediction" in question:
        if current_assessment:
            return (
                "Your demonstration results:\n\n"
                f"- Risk-factor score: {current_assessment.get('risk_factor_score', 'Not available')}\n"
                f"- Risk category: {current_assessment.get('risk_category', 'Not available')}\n"
                f"- Blood pressure category: {current_assessment.get('bp_category', 'Not available')}\n"
                f"- Blood pressure message: {current_assessment.get('bp_message', 'Not available')}\n\n"
                "These results are educational only and are not a diagnosis."
            )

        return "Please complete an assessment first so I can show its results."

    # 6. Fallback
    return (
        "I can answer questions about hypertension, symptoms, risk "
        "factors, blood pressure measurements, lifestyle, treatment "
        "and this VUT Pharmacy demonstration project. Try asking "
        "'What is hypertension?' or 'How does a Decision Tree work?'"
    )


def show_hypertension_chatbot(current_assessment=None):
    st.subheader("💬 VUT Pharmacy Hypertension Assistant")

    st.caption(
        "Ask educational questions about hypertension and the project."
    )

    st.warning(
        "Educational demonstration only. This chatbot cannot diagnose "
        "medical conditions or replace professional medical advice."
    )

    if "hypertension_chat_history" not in st.session_state:
        st.session_state.hypertension_chat_history = []

    for message in st.session_state.hypertension_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a hypertension question...")

    if question:
        st.session_state.hypertension_chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        answer = get_chatbot_answer(
            question,
            current_assessment
        )

        st.session_state.hypertension_chat_history.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

    if st.button("Clear chat"):
        st.session_state.hypertension_chat_history = []
        st.rerun()


