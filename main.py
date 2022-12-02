from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

# Создаем пайплайн для поиска ответа на вопрос в заданном текста.
question_answerer = pipeline('question-answering')


# Создаем свою модель данных = наследуем от BaseModel.
class Item(BaseModel):
    context_question: list


# Создаем приложение FastAPI в переменной app.
app = FastAPI()


#  Определяем функцию и указываем с помощью декоратора что она будет
#  вызываться при GET-запросе (метод HTTP) к корневому каталогу.
@app.get('/')
def root():
    return {'ПРИВЕТСТВИЕ': 'ПРАКТИЧЕСКОЕ ЗАДАНИЕ №3 ПО ДИСЦИПЛИНЕ \"ПРОГРАММНАЯ ИНЖЕНЕРИЯ\"'
                           'УРАЛЬСКОГО ФЕДЕРАЛЬНОГО УНИВЕРСИТЕТА'}


# Определяем функцию и с помощью декоратора указываем что она будет вызываться
# при использовании метода HTTP POST при получении запроса к каталогу /predict.
@app.post('/predict')
def predict(item: Item):
    """
    МОДЕЛЬ ДЛЯ ОТВЕТА НА ВОПРОС ПО ЗАДАННОМУ ТЕКСТУ.
    В PostMan > Body > raw вводим (тип должен быть указан JSON):
        {
            "context_question" : ["My name is Maxim. I am 31 years old.", "What is my name?"]
        }
    """

    result = question_answerer(context=item.context_question[0], question=item.context_question[1])

    # Вычленяем из результата значение оценки.
    score = result['score']
    # Вычленяем из результата собственно сам ответ.
    answer = result['answer']
    # Вычленяем из результата стартовую позицию ответа в заданном тексте.
    start = result['start']
    # Вычленяем из результата конечную позицию ответа в заданном тексте.
    end = result['end']

    # Возвращаем результат в виде словаря.
    return {'SCORE': score,
            'ANSWER': answer,
            'START ANSWER POSITION:': start,
            'END ANSWER POSITION': end
            }
