# frontend/streamlit_app.py
import sys
import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import traceback

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules with error handling
try:
    from src.preprocessing import preprocessor
    from src.model_training import SentimentModel
    from src.utils import load_data
except Exception as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Clothing Reviews Sentiment Analysis",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CACHED LOADING WITH PROGRESS -----------------

@st.cache_resource
def load_model():
    """Load the sentiment analysis model"""
    model = SentimentModel()
    model_path = 'models/logistic_regression.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please run the Jupyter notebook first to train and save the model.")
        return None
    
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file not found: {vectorizer_path}")
        return None
    
    try:
        model.load_model(model_path, vectorizer_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data_cached():
    """Load and prepare the dataset"""
    data_path = 'data/raw/Womens Clothing E-Commerce Reviews.csv'
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}")
        st.info("Please download the dataset from Kaggle and place it in the correct folder.")
        return None
    
    try:
        df = load_data(data_path)
        # Add sentiment column
        df['sentiment'] = df['Rating'].apply(
            lambda x: 'Positive' if x >= 4 else ('Neutral' if x == 3 else 'Negative')
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ----------------- INITIALIZATION WITH PROGRESS BAR -----------------

# Show loading message
with st.spinner("Loading application... Please wait."):
    # Load model and data
    model = load_model()
    df = load_data_cached()
    
    # Check if loading was successful
    if model is None or df is None:
        st.warning("Some components failed to load. Limited functionality available.")
    else:
        st.success("✅ Application loaded successfully!")

# ----------------- SIDEBAR -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA Dashboard", "🤖 Sentiment Analysis", "ℹ️ About"]
)

# ----------------- HOME PAGE -----------------
if page == "🏠 Home":
    st.title("👕 Women's Clothing Reviews Sentiment Analysis")
    st.markdown("---")
    
    if df is not None:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", f"{len(df):,}")
        with col2:
            st.metric("Average Rating", f"{df['Rating'].mean():.2f}/5")
        with col3:
            positive_pct = (df['sentiment'] == 'Positive').mean() * 100
            st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        with col4:
            negative_pct = (df['sentiment'] == 'Negative').mean() * 100
            st.metric("Negative Reviews", f"{negative_pct:.1f}%")
        
        # Quick sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#f39c12',
                'Negative': '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available. Please check the data file.")
    
    st.markdown("---")
    st.subheader("🔍 Try It Yourself")
    
    # Sample reviews for quick testing
    sample_reviews = [
        "I absolutely love this dress! The fabric is amazing and fits perfectly.",
        "The quality is terrible, it fell apart after first wash. Very disappointed.",
        "It's okay, not great but not bad either. Does the job.",
        "Beautiful design and excellent quality! Will buy again."
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sample_text = st.text_area(
            "Enter a review to analyze:",
            sample_reviews[0],
            height=100
        )
    
    with col2:
        st.markdown("### Sample Reviews")
        for i, sample in enumerate(sample_reviews):
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                sample_text = sample
                st.rerun()
    
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if model is not None and hasattr(model, 'is_trained') and model.is_trained:
            with st.spinner("Analyzing..."):
                # Clean and predict
                cleaned = preprocessor.clean_text(sample_text)
                result = model.predict(cleaned)
                
                # Display results
                st.markdown("### Results")
                
                # Sentiment with color
                colors = {
                    'Positive': '#2ecc71',
                    'Neutral': '#f39c12',
                    'Negative': '#e74c3c'
                }
                
                st.markdown(
                    f"<h2 style='color: {colors[result['sentiment']]}; text-align: center;'>"
                    f"{result['sentiment']}</h2>",
                    unsafe_allow_html=True
                )
                
                # Confidence scores
                st.markdown("#### Confidence Scores")
                for sentiment, prob in result['confidence'].items():
                    st.progress(prob, text=f"{sentiment}: {prob:.1%}")
        else:
            st.error("Model not loaded or not trained. Please run the training notebook first.")

# ----------------- EDA DASHBOARD -----------------
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    
    if df is not None:
        # Sample data for faster plotting
        if len(df) > 5000:
            plot_df = df.sample(5000, random_state=42)
            st.info(f"Showing sample of 5000 reviews (total: {len(df):,})")
        else:
            plot_df = df
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Distributions",
            "📊 Ratings Analysis",
            "👥 Demographics",
            "📝 Text Analysis"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution")
                sentiment_counts = plot_df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                fig = px.bar(
                    sentiment_counts,
                    x='Sentiment',
                    y='Count',
                    color='Sentiment',
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#f39c12',
                        'Negative': '#e74c3c'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Rating Distribution")
                rating_counts = plot_df['Rating'].value_counts().sort_index().reset_index()
                rating_counts.columns = ['Rating', 'Count']
                fig = px.bar(
                    rating_counts,
                    x='Rating',
                    y='Count',
                    color='Rating',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Rating vs Age")
                fig = px.scatter(
                    plot_df.sample(min(1000, len(plot_df))),
                    x='Age',
                    y='Rating',
                    color='sentiment',
                    opacity=0.6,
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#f39c12',
                        'Negative': '#e74c3c'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Recommendation Rate by Rating")
                rec_by_rating = plot_df.groupby('Rating')['Recommended IND'].mean() * 100
                fig = px.line(
                    x=rec_by_rating.index,
                    y=rec_by_rating.values,
                    markers=True,
                    labels={'x': 'Rating', 'y': 'Recommendation %'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Age Distribution")
                fig = px.histogram(
                    plot_df,
                    x='Age',
                    color='sentiment',
                    nbins=30,
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#f39c12',
                        'Negative': '#e74c3c'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Department Performance")
                if 'Department Name' in plot_df.columns:
                    dept_sentiment = pd.crosstab(
                        plot_df['Department Name'],
                        plot_df['sentiment'],
                        normalize='index'
                    ) * 100
                    fig = px.bar(
                        dept_sentiment.reset_index(),
                        x='Department Name',
                        y=['Positive', 'Neutral', 'Negative'],
                        title='Sentiment by Department',
                        barmode='stack',
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Neutral': '#f39c12',
                            'Negative': '#e74c3c'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Coming Soon: Word Cloud Analysis")
            st.info("Word cloud feature will be added in the next update.")
    else:
        st.warning("No data available. Please check the data file.")

# ----------------- BATCH SENTIMENT ANALYSIS -----------------
elif page == "🤖 Sentiment Analysis":
    st.title("🤖 Batch Sentiment Analysis")
    
    if model is not None and hasattr(model, 'is_trained') and model.is_trained:
        st.markdown("Upload a CSV file containing reviews to analyze them in batch.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File must contain a column named 'Review Text'"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                input_df = pd.read_csv(uploaded_file)
                st.write(f"📄 File loaded: {len(input_df)} rows")
                
                # Check for required column
                if 'Review Text' in input_df.columns:
                    st.write("Preview of uploaded data:")
                    st.dataframe(input_df[['Review Text']].head())
                    
                    # Analysis button
                    if st.button("🔍 Analyze All Reviews", type="primary", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        total = len(input_df)
                        
                        for i, review in enumerate(input_df['Review Text'].dropna()):
                            # Update progress
                            progress = (i + 1) / total
                            progress_bar.progress(progress)
                            status_text.text(f"Processing review {i+1} of {total}...")
                            
                            # Analyze
                            cleaned = preprocessor.clean_text(review)
                            result = model.predict(cleaned)
                            
                            results.append({
                                'Review': review[:100] + "..." if len(review) > 100 else review,
                                'Sentiment': result['sentiment'],
                                'Positive_Confidence': f"{result['confidence']['Positive']:.2%}",
                                'Neutral_Confidence': f"{result['confidence']['Neutral']:.2%}",
                                'Negative_Confidence': f"{result['confidence']['Negative']:.2%}"
                            })
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        results_df = pd.DataFrame(results)
                        st.success(f"✅ Analysis complete! Processed {len(results)} reviews.")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.error("❌ CSV file must contain a column named 'Review Text'")
                    st.info("Please make sure your CSV has a column with exactly this name: 'Review Text'")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.exception(e)
    else:
        st.warning("⚠️ Model not loaded. Please run the training notebook first to create the model.")
        st.info("Run the Jupyter notebook '01_sentiment_analysis_complete.ipynb' to train and save the model.")

# ----------------- ABOUT PAGE -----------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Women's Clothing Reviews Sentiment Analysis
        
        This project uses Natural Language Processing (NLP) and Machine Learning to automatically
        classify customer reviews into Positive, Neutral, or Negative sentiments.
        
        **Features:**
        - 📊 Interactive EDA dashboard
        - 🤖 Real-time sentiment prediction
        - 📈 Batch processing of multiple reviews
        - 🎨 Beautiful visualizations
        
        **Technical Stack:**
        - Python 3.10+
        - Streamlit for web interface
        - Scikit-learn for ML models
        - NLTK for text preprocessing
        - Plotly for interactive charts
        - Pandas for data manipulation
        
        **Model Performance:**
        - Accuracy: ~90%
        - Best Model: Logistic Regression with TF-IDF
        - Features: 5000 most important words and phrases
        
        **Developer:** Warda Adan
        
        **Links:**
        - [GitHub](https://github.com/wardaadan03-netizen)
        - [LinkedIn](https://www.linkedin.com/in/thewardaadan-wa/)
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/96/000000/fashion.png", width=150)
        st.markdown("### Quick Stats")
        if df is not None:
            st.metric("Dataset Size", f"{len(df):,} reviews")
            st.metric("Model Accuracy", "89.5%")
            st.metric("Python Version", "3.10+")

# ----------------- FOOTER -----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
if model is not None and hasattr(model, 'is_trained') and model.is_trained:
    st.sidebar.success("✅ Model: Loaded")
else:
    st.sidebar.error("❌ Model: Not Loaded")

if df is not None:
    st.sidebar.success(f"✅ Data: {len(df):,} reviews")
else:
    st.sidebar.error("❌ Data: Not Loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")