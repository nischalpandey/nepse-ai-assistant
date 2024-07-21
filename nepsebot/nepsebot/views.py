from rest_framework import viewsets
from rest_framework import response
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import UserQuery
from .serializer import UserQuerySerializer
from nepsebot.ai.main import process_user_query, ConversationModel 


from django.http import HttpResponse
from django.template import loader
import os
def index(request):
  template = loader.get_template('index.html')
  return HttpResponse(template.render())


  
class QueryView(APIView):
    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai/mi_model/nepse_model_improved.pkl')
            cls.model = ConversationModel.load_model(path)
        return cls.model

    def post(self, request):
        user_input = request.data.get('query')
        if not user_input:
            return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)

        conversation_model = self.get_model()
        response = process_user_query(user_input, conversation_model)

        # Save the query and response
        query = UserQuery(query=user_input, response=response)
        query.save()

        serializer = UserQuerySerializer(query)
        return Response(serializer.data, status=status.HTTP_201_CREATED)