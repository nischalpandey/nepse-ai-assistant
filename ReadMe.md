# NEPSE AI Assistant

An intelligent virtual assistant and customer support bot powered by Python.

## Features


1. **Nepal Stock Exchange (NEPSE) Knowledge Base**
   - Comprehensive information about NEPSE
   - Stock data integration

2. **Stock Price Information**
   - Latest price updates for NEPSE stocks
   - Historical price data and trends

3. **Natural Language Query Processing**
   - Advanced NLP , understanding user queries
   - Contextual response generation

4. **User Information Management**
   - Secure storage and retrieval of user data
   - Personalized responses to user information requests

5. **AI-Powered Predictions and Analysis**
   - Stock market trend analysis
   - Predictive modeling for investment insights

## Technology Stack
- **Backend**: Django


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/nischalpandey/nepse-ai-assistant.git
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Change Dir:
   ```
   cd nepsebot
   ```
5. Set up the database:
   ```
   python manage.py migrate
   ```

6. Run the development server:
   ```
   python manage.py runserver
   ```

## Usage

1. Start the Django server
2. Access the bot through the provided web interface or API endpoints
3. Begin interacting with the AI assistant using natural language queries

## Configuration

1. Set up your environment variables in a `.env` file:
   ```
   DEBUG=True
   SECRET_KEY=your_secret_key
   DATABASE_URL=your_database_url
   ```



## TODO List

- [ ] Implement user authentication system
- [ ] Train the Chat Model in Large DataSet
- [ ] Integrate real-time NEPSE data feed
- [ ] Develop advanced NLP models for Nepali language support
- [ ] Create a dashboard for data visualization
- [ ] Set up manual or automated testing for NLP and AI components
- [ ] Implement voice recognition for query input


## Contributing

Contributions are welcomed anytime!. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Django community for their excellent documentation
- Open-source AI libraries that make this project possible

For any questions or support, please open an issue or contact our team at hello@nischalpandey.com.np.
