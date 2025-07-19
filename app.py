import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import BaseOutputParser
import json
from datetime import datetime
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Custom output parser for structured resume data
class ResumeOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to simple text parsing
                return {"content": text.strip()}
        except:
            return {"content": text.strip()}

# Initialize LangChain components
@st.cache_resource
def init_langchain():
    # Get API key from multiple sources (priority order)
    api_key = (
        st.session_state.get("openai_api_key", "") or  # From sidebar input
        os.getenv("OPENAI_API_KEY", "") or             # From .env file
        st.secrets.get("OPENAI_API_KEY", "")           # From streamlit secrets
    )
    
    if not api_key:
        st.error("❌ OpenAI API key not found! Please:")
        st.error("1. Enter your API key in the sidebar, OR")
        st.error("2. Add OPENAI_API_KEY to your .env file, OR") 
        st.error("3. Add it to .streamlit/secrets.toml")
        return None, None
    
    try:
        # Initialize OpenAI LLM
        llm = OpenAI(temperature=0.7, api_key=api_key)
        st.success("✅ OpenAI API key loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing OpenAI: {str(e)}")
        return None, None
    
    # Resume generation prompt
    resume_prompt = PromptTemplate(
        input_variables=["name", "email", "phone", "experience", "education", "skills", "job_type"],
        template="""
        Create a professional resume for the following person:
        
        Name: {name}
        Email: {email}
        Phone: {phone}
        Target Job Type: {job_type}
        
        Experience:
        {experience}
        
        Education:
        {education}
        
        Skills:
        {skills}
        
        Please create a well-formatted, professional resume that highlights the most relevant experiences and skills for the {job_type} position. 
        Include appropriate sections like Summary, Experience, Education, Skills, and any other relevant sections.
        Format it in a clean, professional manner.
        """
    )
    
    # Resume improvement prompt
    improvement_prompt = PromptTemplate(
        input_variables=["resume", "job_description"],
        template="""
        Please improve the following resume to better match this job description:
        
        Job Description:
        {job_description}
        
        Current Resume:
        {resume}
        
        Please provide specific suggestions for improvement and a revised version that better aligns with the job requirements.
        Focus on relevant keywords, skills, and experiences that match the job description.
        """
    )
    
    # Create chains
    resume_chain = LLMChain(llm=llm, prompt=resume_prompt)
    improvement_chain = LLMChain(llm=llm, prompt=improvement_prompt)
    
    return resume_chain, improvement_chain

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Resume Builder", page_icon="📄", layout="wide")
    
    st.title("🤖 AI Resume Builder")
    st.markdown("Build professional resumes with AI assistance using LangChain and OpenAI")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Create Resume", "Improve Resume", "Resume Templates"])
    
    if page == "Create Resume":
        create_resume_page()
    elif page == "Improve Resume":
        improve_resume_page()
    else:
        templates_page()

def create_resume_page():
    st.header("📝 Create New Resume")
    
    # Personal Information
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name", placeholder="John Doe")
        email = st.text_input("Email", placeholder="john.doe@email.com")
    
    with col2:
        phone = st.text_input("Phone Number", placeholder="+1 (555) 123-4567")
        job_type = st.text_input("Target Job Position", placeholder="Software Engineer")
    
    # Experience Section
    st.subheader("Professional Experience")
    experience = st.text_area(
        "Describe your work experience (one job per line)",
        placeholder="Software Engineer at Tech Corp (2020-2023): Developed web applications using Python and React\nIntern at StartupXYZ (2019-2020): Assisted in mobile app development",
        height=150
    )
    
    # Education Section
    st.subheader("Education")
    education = st.text_area(
        "Educational background",
        placeholder="Bachelor of Science in Computer Science, University of Technology (2016-2020)\nRelevant Coursework: Data Structures, Algorithms, Database Systems",
        height=100
    )
    
    # Skills Section
    st.subheader("Skills")
    skills = st.text_area(
        "List your skills (separate with commas)",
        placeholder="Python, JavaScript, React, Node.js, SQL, Git, Docker, AWS",
        height=100
    )
    
    # Generate Resume Button
    if st.button("🚀 Generate Resume", type="primary"):
        # Debug: Show what values we have
        st.write("Debug - Values received:")
        st.write(f"Name: '{name}' (length: {len(name) if name else 0})")
        st.write(f"Email: '{email}' (length: {len(email) if email else 0})")
        st.write(f"Phone: '{phone}' (length: {len(phone) if phone else 0})")
        
        # Check if required fields are filled (strip whitespace)
        if not name.strip() or not email.strip() or not phone.strip():
            st.error("Please fill in all required personal information fields")
            st.error(f"Missing: {', '.join([field for field, value in [('Name', name.strip()), ('Email', email.strip()), ('Phone', phone.strip())] if not value])}")
            return
        
        try:
            with st.spinner("Generating your resume..."):
                chains = init_langchain()
                if chains[0] is None:
                    return
                
                resume_chain, _ = chains
                
                # Generate resume using LangChain
                result = resume_chain.invoke({
                    "name": name.strip(),
                    "email": email.strip(),
                    "phone": phone.strip(),
                    "experience": experience.strip() if experience else "No experience provided",
                    "education": education.strip() if education else "No education provided",
                    "skills": skills.strip() if skills else "No skills provided",
                    "job_type": job_type.strip() if job_type else "General"
                })
                
                # Extract the content from the result
                resume_content = result["text"] if isinstance(result, dict) and "text" in result else str(result)
                
                # Display generated resume
                st.success("Resume generated successfully!")
                st.subheader("Generated Resume")
                st.text_area("Your Resume", resume_content, height=400)
                
                # Download button
                st.download_button(
                    label="📄 Download Resume",
                    data=resume_content,
                    file_name=f"{name.replace(' ', '_')}_resume.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Error generating resume: {str(e)}")
            st.info("Make sure you have set up your OpenAI API key in Streamlit secrets")

def improve_resume_page():
    st.header("✨ Improve Existing Resume")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Resume")
        current_resume = st.text_area(
            "Paste your current resume here",
            placeholder="Paste your existing resume content...",
            height=300
        )
    
    with col2:
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the job description you're targeting",
            placeholder="Paste the job description here...",
            height=300
        )
    
    if st.button("🔄 Improve Resume", type="primary"):
        if not current_resume or not job_description:
            st.error("Please provide both your current resume and the job description")
            return
        
        try:
            with st.spinner("Analyzing and improving your resume..."):
                chains = init_langchain()
                if chains[0] is None:
                    return
                
                _, improvement_chain = chains
                
                # Improve resume using LangChain
                result = improvement_chain.invoke({
                    "resume": current_resume.strip(),
                    "job_description": job_description.strip()
                })
                
                # Extract the content from the result
                improved_content = result["text"] if isinstance(result, dict) and "text" in result else str(result)
                
                # Display improved resume
                st.success("Resume improved successfully!")
                st.subheader("Improved Resume & Suggestions")
                st.text_area("Improved Resume", improved_content, height=400)
                
                # Download button
                st.download_button(
                    label="📄 Download Improved Resume",
                    data=improved_content,
                    file_name=f"improved_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Error improving resume: {str(e)}")
            st.info("Make sure you have set up your OpenAI API key in Streamlit secrets")

def templates_page():
    st.header("📋 Resume Templates & Tips")
    
    st.subheader("Resume Writing Tips")
    tips = [
        "Use action verbs like 'developed', 'managed', 'led', 'implemented'",
        "Quantify your achievements with numbers and percentages",
        "Tailor your resume to each job application",
        "Keep it concise - ideally 1-2 pages",
        "Use a clean, professional format",
        "Include relevant keywords from the job description",
        "Proofread for spelling and grammar errors"
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")
    
    st.subheader("Sample Resume Sections")
    
    with st.expander("Professional Summary Example"):
        st.markdown("""
        **Professional Summary:**
        Experienced Software Engineer with 5+ years of expertise in full-stack development, 
        specializing in Python, React, and cloud technologies. Proven track record of delivering 
        scalable web applications and leading cross-functional teams. Passionate about clean code 
        and continuous learning.
        """)
    
    with st.expander("Experience Section Example"):
        st.markdown("""
        **Senior Software Engineer** | Tech Solutions Inc. | 2021 - Present
        • Developed and maintained 3 high-traffic web applications serving 100K+ users
        • Led a team of 4 developers in implementing new features using Agile methodologies
        • Reduced application load time by 40% through code optimization and caching strategies
        • Collaborated with product managers and designers to deliver user-centered solutions
        """)
    
    with st.expander("Skills Section Example"):
        st.markdown("""
        **Technical Skills:**
        • Programming Languages: Python, JavaScript, TypeScript, Java
        • Frameworks: React, Node.js, Django, Flask
        • Databases: PostgreSQL, MongoDB, Redis
        • Tools: Git, Docker, AWS, Jenkins
        • Methodologies: Agile, Test-Driven Development
        """)

# Configuration sidebar
def setup_sidebar():
    st.sidebar.markdown("---") 
    st.sidebar.subheader("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="api_key_input")
    if api_key:
        st.session_state.openai_api_key = api_key
        st.sidebar.success("API Key set successfully!")
    
    # Clear cache button
    if st.sidebar.button("Clear Cache"):
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 How to Use")
    st.sidebar.markdown("""
    1. **Create Resume**: Fill in your information and generate a new resume
    2. **Improve Resume**: Upload existing resume and job description for optimization
    3. **Templates**: View examples and tips for better resumes
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 Setup")
    st.sidebar.markdown("""
    To use this app, you need an OpenAI API key. 
    Get one at: https://platform.openai.com/api-keys
    """)

if __name__ == "__main__":
    setup_sidebar()
    main()