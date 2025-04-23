import streamlit as st
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="ExciumEdu", layout="wide")

# ------------------ Setup ------------------
@st.cache_resource(show_spinner="Loading resources...")
def setup_bot():
    loader = DirectoryLoader("data", glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")

    contextual_prompt = ChatPromptTemplate.from_messages([
        ("system", "Turn the latest user message into a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using this context: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware = create_history_aware_retriever(llm, retriever, contextual_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware, qa_chain)

    memory_store = {}
    def get_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in memory_store:
            memory_store[session_id] = ChatMessageHistory()
        return memory_store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

bot_chain = setup_bot()

# ------------------ Predefined Q&A ------------------
predefined_qna = {
    "Basic Introduction": [
        ("Hi, how are you?", "Hello! I'm doing well, thank you for asking. I'm here to help you with any questions about our educational programs. How can I assist you today?"),
        ("Who are you?", "I'm your virtual educational counselor, designed to provide information about our college programs, admissions, fees, and other student services."),
        ("How can you help me?", "I can provide information about our courses, admission requirements, deadlines, fees, scholarships, and more."),
        ("Hi, how can I help you with your educational journey today?", "Hello! I'm your educational counseling assistant. I can provide information about programs, admissions, financial aid, and campus life. What specific information are you looking for today?"),
        ("What educational programs does your institution offer?", "We offer a wide range of programs including undergraduate degrees (Bachelor's), graduate programs (Master's and PhD), certificate programs, and continuing education courses across fields like Business, Engineering, Arts, Sciences, Healthcare, and Education. Would you like details about any specific area?"),
        ("How do I know which program is right for me?", "Choosing the right program depends on your interests, career goals, academic background, and personal circumstances. I recommend exploring our program catalog, attending virtual information sessions, or scheduling a one-on-one consultation with an academic advisor who can help match your goals with appropriate programs."),
        ("What are the admission requirements for undergraduate programs?", "For undergraduate programs, we typically require a high school diploma or equivalent, a minimum GPA of 3.0, standardized test scores (SAT/ACT, though some programs are test-optional), a personal statement, and sometimes letters of recommendation. Specific programs may have additional requirements like portfolios for art programs or prerequisites for science programs."),
        ("When are the application deadlines?", "Our main application deadlines are: Early Decision - November 1, Regular Decision - January 15 for Fall semester, and October 1 for Spring semester. Some graduate and specialized programs may have different deadlines, so I recommend checking the specific program page on our website."),
        ("What financial aid and scholarship opportunities are available?", "We offer merit-based scholarships, need-based grants, work-study opportunities, and various external scholarship connections. To be considered, complete the FAFSA (Free Application for Federal Student Aid) and our institutional scholarship application by March 1 for priority consideration."),
        ("How much is tuition and what other costs should I expect?", "Undergraduate tuition is approximately $35,000 per academic year. Additional costs include housing ($10,000-14,000/year), meal plans ($4,000-6,000/year), books and supplies (approximately $1,200/year), and student fees ($1,500/year). Financial aid packages can significantly reduce these costs for eligible students."),
        ("Do you offer online or hybrid learning options?", "Yes, we have expanded our flexible learning options. Many programs offer fully online, hybrid, and evening/weekend formats to accommodate diverse student needs. Our learning management system provides comprehensive support for distance learners, including virtual office hours with professors."),
        ("What career services do you provide for students?", "Our Career Development Center offers career counseling, resume and interview workshops, job fairs, internship placement assistance, networking events with alumni, and a job portal exclusive to our students and graduates. These services continue to be available to alumni after graduation."),
        ("How can I schedule a campus tour or speak with an advisor?", "You can schedule a campus tour through our website's 'Visit Us' section. Virtual tours are also available. To speak with an admissions advisor, you can book an appointment online, call our admissions office at (555) 123-4567, or email admissions@university.edu. We offer both in-person and virtual advising sessions.")
    ],
        "Course Information": [
        ("What courses does the college provide?", "Our college offers Business, CS, Engineering, Nursing, Psychology, Education, Design, and Culinary Arts."),
        ("What is the duration of the Bachelor's programs?", "Most Bachelor's degrees are 4 years (8 semesters), with some variations."),
        ("Do you offer any short-term courses?", "Yes, certificate and diploma courses from 3 months to 1 year."),
        ("Are there any online programs available?", "Yes, we offer fully online and hybrid programs in Business, IT, and Education."),
        ("What specializations are available in Engineering?", "Specializations include Civil, Mechanical, Electrical, Computer, and Chemical Engineering."),
        ("What undergraduate majors are most popular at your institution?", "Our most popular undergraduate majors include Business Administration, Computer Science, Psychology, Biological Sciences, and Engineering. These programs have excellent faculty, strong industry connections, and high graduate employment rates. Would you like more specific information about any of these fields?"),
        ("How many credit hours are typically required to complete a Bachelor's degree?", "Most of our Bachelor's degree programs require 120 credit hours to complete. This typically translates to about 40 courses over four years. Some specialized programs, particularly in Engineering or Architecture, may require up to 128-136 credit hours due to additional technical requirements."),
        ("Do you offer any accelerated degree programs?", "Yes, we offer several accelerated programs that allow students to complete their degrees more quickly. Our 3+1 programs let students earn a Bachelor's and Master's degree in four years, and our fast-track options allow motivated students to complete a standard Bachelor's degree in three years through summer courses and higher credit loads per semester."),
        ("What is the average class size for undergraduate courses?", "Our average undergraduate class size is 27 students. Introductory courses may be larger (around 50-100 students) but include smaller discussion sections led by teaching assistants. Upper-level courses are much smaller, typically 15-20 students, allowing for more personalized instruction and meaningful discussions."),
        ("Are there opportunities for undergraduate research?", "Absolutely! We strongly encourage undergraduate research across all disciplines. Our Undergraduate Research Program connects students with faculty mentors, provides research grants, and hosts an annual symposium where students present their work. Many students co-author publications and present at national conferences."),
        ("What graduate programs do you offer?", "We offer over 75 graduate programs including Master's degrees, PhDs, and professional doctorates. These span fields such as Business (MBA), Education (MEd, EdD), Engineering (MS, PhD), Health Sciences (MPH, MSN), Computer Science (MS), and Arts & Humanities (MA, MFA). Many programs offer both full-time and part-time options."),
        ("How are courses structured? Do you use semesters or quarters?", "We operate on a semester system with Fall (August-December) and Spring (January-May) terms of 15 weeks each, plus a Summer term with multiple sessions of varying lengths. Most courses meet 2-3 times per week, though laboratory components, studio classes, and seminars may have different scheduling patterns."),
        ("Do you offer interdisciplinary programs or the ability to design my own major?", "Yes, we offer several established interdisciplinary programs such as Environmental Studies, Digital Media, and Global Health. Additionally, our Individualized Studies program allows motivated students to design their own major with faculty guidance, combining courses from different departments to create a unique educational path aligned with specific career goals."),
        ("What internship or cooperative education opportunities are available as part of the curriculum?", "Many of our programs integrate internship experiences into the curriculum, with some majors requiring internships for graduation. Our cooperative education program allows students to alternate semesters of full-time study with full-time paid work in their field. These experiences are credit-bearing and supervised by faculty to ensure educational quality."),
        ("Are there study abroad opportunities, and how do they fit into degree programs?", "We have partnerships with over 100 universities worldwide, offering semester, year-long, and short-term study abroad opportunities. Most programs are designed to integrate seamlessly with degree requirements, allowing students to take major courses abroad without delaying graduation. Scholarships are available to support international experiences.")
    ],
        "Admissions": [
        ("What is the application process for undergraduate admissions?",
        "Submit the online application, official high school transcripts, SAT/ACT scores (optional), a personal statement, and pay the application fee. Some programs may require portfolios or auditions."),

        ("How competitive is the admissions process?",
        "We have an acceptance rate of about 65%. We consider GPA, test scores, extracurriculars, essays, and personal achievements in a holistic review."),

        ("Do you require letters of recommendation?",
        "Yes. We recommend one letter from a teacher and another from a counselor or non-academic mentor like a coach or employer."),

        ("Is an interview required?",
        "Interviews are optional but recommended. Some programs (e.g., Nursing, Business Honors) require them. Interviews can be online or in-person."),

        ("What is your policy on transferring credits from other institutions?",
        "We accept transfer credits with a grade of C or better from accredited institutions. Up to 60 credits may be transferred. Official transcripts are required."),

        ("Do you offer early decision or early action?",
        "Yes. Early Decision (binding) and Early Action (non-binding) are both available. Deadlines: November 1. Regular Decision: February 1."),

        ("What are the admission requirements for international students?",
        "International applicants must submit English proficiency scores (TOEFL, IELTS, or Duolingo), financial documents, and credential evaluations."),

        ("How do I apply for graduate programs?",
        "Apply online. Requirements typically include a bachelor's degree, transcripts, letters of recommendation, statement of purpose, resume/CV, and possibly GRE/GMAT or a portfolio."),

        ("Do you offer application fee waivers?",
        "Yes, for eligible students with financial need. Fee waivers are available for students in programs like TRIO or with SAT/ACT fee waivers."),

        ("What factors are considered in the holistic review?",
        "We consider GPA, test scores, course rigor, extracurriculars, essays, leadership, and potential contributions to the campus community.")
    ],
    "Fees and Financial Aid": [
        ("How much is the registration fee?", "$50 for domestic, $75 for international. Non-refundable."),
        ("What is the tuition fee per semester?", "Around $12,000 for undergrad; $15k–$20k for grad programs."),
        ("Are there any scholarships available?", "Yes, merit-based, need-based, athletic, and program-specific."),
        ("How can I apply for financial aid?", "Submit FAFSA and institutional aid application."),
        ("Are there payment plans available?", "Yes, monthly plans with a $25 setup fee per semester.")
    ],
    "Campus Life": [
        ("What housing options are available for students on campus?",
        "We offer traditional dorms, suite-style and apartment-style housing, and themed living-learning communities. First-year students are guaranteed housing if they apply by May 1."),

        ("What dining options are available on campus?",
        "There are 12 dining locations including dining halls, food courts, coffee shops, and international cuisine options. Meal plans are customizable and support dietary needs like vegan, gluten-free, and halal."),

        ("What student organizations and clubs can I join?",
        "We have over 250 student organizations, from academic and cultural groups to performance arts, service orgs, and student government. You can even start your own club with 5 students and a faculty advisor."),

        ("What recreational and fitness facilities are available?",
        "The Rec Center includes a fitness center, pools, courts, indoor track, and group fitness studios. Students can join intramurals, outdoor trips, and use personal training services—all included in student fees."),

        ("Is the campus safe, and what security measures are in place?",
        "Yes, safety is a top priority. We have 24/7 campus police, blue light emergency phones, escort services, security cameras, and card-access residence halls. Our alert system sends notifications via text, email, and app."),

        ("What mental health and wellness resources are available?",
        "Students have access to 12 free counseling sessions annually, group therapy, 24/7 helpline, psychiatric care, mindfulness workshops, and peer support—all provided confidentially by licensed professionals."),

        ("What transportation options exist on and around campus?",
        "Free shuttles run weekdays and weekends. There's a bike-share program, free city bus access for students, and various parking permits. First-year residents may have restrictions on bringing cars."),

        ("What arts and cultural activities happen on campus?",
        "Campus hosts theater, dance, music concerts, galleries, film screenings, and festivals. Students enjoy discounted or free entry, and we have an artist-in-residence program with workshops and special events."),

        ("What academic support services are available outside the classroom?",
        "The Academic Success Center offers tutoring, writing support, study skills workshops, coaching, and help for first-generation students—available in person and online with flexible hours."),

        ("How diverse is the campus community, and what inclusion initiatives exist?",
        "Our community includes students from all 50 states and many countries. We support diversity through cultural centers, affinity groups, DEI programming, and inclusive leadership development initiatives.")
    ], 
    "Student General Counseling": [
        ("I'm feeling overwhelmed with my coursework. What should I do?",
        "Start by breaking down your workload into manageable tasks and creating a weekly schedule. Visit our Academic Support Center for time management workshops and study skills coaching. If stress is affecting your well-being, our Counseling Center offers stress management sessions and short-term counseling. Prioritize self-care—adequate sleep, nutrition, and breaks can improve focus and productivity."),

        ("How can I balance my academic responsibilities with extracurricular activities?",
        "Use a planner or digital calendar to map out all commitments. Limit non-academic work to 15-20 hours weekly. Prioritize activities that align with your goals. Our Student Success Coaches can help you develop strategies for balance, so you can enjoy a well-rounded college experience."),

        ("I'm struggling to decide on a major. What resources can help me?",
        "Our Career Development Center offers major and career assessments. You can enroll in our 'Major Exploration' course and attend departmental sessions. Meet with faculty and academic advisors to guide you in selecting a major that fits your interests and career aspirations."),

        ("I'm experiencing conflict with my roommate. How should I handle this?",
        "Start by having a calm, direct conversation with your roommate about the issues. Revisit your roommate agreement if needed. If issues persist, contact your Resident Advisor to facilitate a mediation. For serious conflicts, Residence Life staff can help with room changes."),

        ("I'm feeling homesick and having trouble making friends. What can I do?",
        "Join residence hall events, attend student organizations, and consider our Peer Connection program for mentorship. Homesickness is normal—our Counseling Center offers group counseling for first-year students. Friendships take time to develop, so give yourself grace."),

        ("How can I manage test anxiety?",
        "Practice relaxation techniques such as progressive muscle relaxation and deep breathing. Our Academic Support Center offers workshops for test anxiety. Prepare thoroughly using study guides and practice tests. If needed, our Counseling Center provides individual and group therapy for test anxiety."),

        ("I think I might have a learning disability. What should I do?",
        "Meet with our Disability Services Office for a confidential consultation. If necessary, they can refer you for formal evaluation. If diagnosed, you may qualify for accommodations such as extended time on exams or note-taking support.")
    ]
}


# ------------------ Sidebar ------------------
# Persistent toggle with checkbox to show history
# st.session_state.show_history = st.sidebar.checkbox("🕘 Show History", value=st.session_state.get("show_history", False))

# # Expander to simulate a pop-up window in the main panel
# if st.session_state.show_history:
#     with st.expander("📝 Conversation History"):
#         # Check if there's any history saved in session state
#         if "history" in st.session_state and st.session_state.history:
#             for role, msg in st.session_state.history:
#                 st.markdown(f"**{role}:** {msg}")
#         else:
#             st.markdown("No history yet.")

# Persistent toggle with checkbox to show history
st.session_state.show_history = st.sidebar.checkbox("🕘 Show History", value=st.session_state.get("show_history", False))

# Expander to simulate a pop-up window in the main panel
if st.session_state.show_history:
    with st.expander("📝 Conversation History"):
        # Check if there's any history saved in session state
        if "history" in st.session_state and st.session_state.history:
            for role, msg in st.session_state.history:
                st.markdown(f"**{role}:** {msg}")
        else:
            st.markdown("No history yet.")

st.sidebar.title("📚 ExciumEdu Assistant Options")
category = st.sidebar.selectbox("Choose a category", list(predefined_qna.keys()))
st.sidebar.markdown("### Predefined Questions")
for q, _ in predefined_qna[category]:
    if st.sidebar.button(q):
        st.session_state.input = q

# ------------------ Chat UI ------------------
st.title("🎓 ExciumEdu: Your Educational Assistant")
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Type your message here:", value=st.session_state.get("input", ""), key="user_input")
# st.session_state.input = ""  # Clear after using

if user_input.strip():
    st.session_state.history.append(("You", user_input))
    matched = False

    for q, a in predefined_qna.get(category, []):
        if user_input.strip().lower() == q.strip().lower():
            st.write("EduMind:", a)
            st.session_state.history.append(("EduMind", a))
            matched = True
            break


    if not matched:
        with st.spinner("Thinking..."):
            response = bot_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "user"}}
            )
            answer = response.get("answer", "I'm not sure how to answer that.")
            st.write("EduMind:", answer)
            st.session_state.history.append(("EduMind", answer))
            
