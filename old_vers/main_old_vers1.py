import logging
from bs4 import BeautifulSoup
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils import executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import nn_cllassifier
import requests
from nn_cllassifier import preprocess_text, load_own_model, predict
import model_path
import data_ready
from data_ready import class_answers, LabelDict
import numpy as np
import ner_recogn
from ner_recogn import extract_entities
import parser_shedule
from parser_shedule import get_group_schedule_link, get_teacher_page_url, get_teacher_schedule_links
API_TOKEN = '6501382603:AAGYDgjoV718Z_1p7PSViCRI8H4zW7RnyB0'

logging.basicConfig(level=logging.INFO)

storage = MemoryStorage()
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=storage)
dp.middleware.setup(LoggingMiddleware())
selected_faculty_link = None
# для единоразовой загрузки модели при старте бота:
tokenizer = None
model = None
# флаг что модель была успешно загружена
load_model_success = False

# Определение клавиатуры с кнопкой /start
start_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
start_button = types.KeyboardButton('/start')
start_keyboard.add(start_button)

# Функция для парсинга факультетов с сайта
def parse_faculties():
    url = "https://www.sgu.ru/schedule"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    faculties = {}
    for link in soup.select('a[href*="/schedule/"]'):
        faculty_name = ' '.join([word[0].upper() for word in link.text.strip().split()])
        faculty_link = "https://www.sgu.ru" + link['href']
        faculties[faculty_name] = faculty_link
    
    return faculties
# Создание клавиатуры с факультетами для текущей страницы
def create_faculties_keyboard(faculties, page=0, items_per_page=4):
    keyboard = InlineKeyboardMarkup()
    faculties_list = list(faculties.items())
    start = page * items_per_page
    end = start + items_per_page
    for faculty_name, faculty_link in faculties_list[start:end]:
        keyboard.add(InlineKeyboardButton(text=faculty_name, callback_data=f"faculty_{faculty_name}"))
    
    navigation_buttons = []
    if start > 0:
        navigation_buttons.append(InlineKeyboardButton(text="Previous", callback_data=f"prev_{page-1}"))
    if end < len(faculties):
        navigation_buttons.append(InlineKeyboardButton(text="Next", callback_data=f"next_{page+1}"))
    
    if navigation_buttons:
        keyboard.row(*navigation_buttons)
    
    return keyboard
# Клавиатура с ответами "Да" и "Нет"
def get_confirm_keyboard():
    confirm_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    yes_button = types.KeyboardButton('Да')
    no_button = types.KeyboardButton('Нет')
    confirm_keyboard.add(yes_button, no_button)
    return confirm_keyboard

class Form(StatesGroup):
    confirm_response = State()

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    print("bot started")
    global tokenizer
    global model
    global load_model_success
    # Загружаем модель и токенизатор один раз
    if tokenizer is None or model is None:
        model_path_dir = model_path.model_path
        tokenizer, model = load_own_model(model_path_dir)
        model.eval()  # Перевод модели в режим оценки
        if not (tokenizer is None) and not (model is None):
            load_model_success = True
            print("model loaded")
    
    # Приветственное сообщение
    await message.reply("Привет! Я СГУшка: Бот-помощник. Отправь мне сообщение с вопросом, и я постараюсь помочь.", reply_markup=start_keyboard)

@dp.message_handler(lambda message: message.text not in ["Да", "Нет"], state='*')
async def handle_message(message: types.Message, state: FSMContext):
    if not load_model_success:
        await message.reply("Модель не загружена. Попробуйте позже.")
        return
    
    user_text = message.text
    probabilities = predict(user_text, tokenizer, model)

    if np.max(probabilities) <= 1e-3:
        broken_response = class_answers['uknow']['text']
        await message.reply(broken_response)
    else:
        top_indices = np.argsort(probabilities[0])[::-1]
        current_attempt = 0

        await send_response(message, state, top_indices, probabilities[0], current_attempt)

async def send_response(message, state: FSMContext, top_indices, probabilities, attempt):
    # Добавляем отладочные сообщения для проверки массивов
    logging.info(f"Attempt: {attempt}")
    logging.info(f"Top indices: {top_indices}")
    logging.info(f"Probabilities: {probabilities}")
    
    # Проверяем длину top_indices и наличие необходимого индекса
    if attempt < min(2, len(top_indices)) and probabilities[top_indices[attempt]] >= 1e-4:
        predicted_class_index = top_indices[attempt]
        predicted_class = LabelDict[predicted_class_index]
        response_text = class_answers[predicted_class]["text"]
        user_text = message.text
        #print(, ' - сообщение?')
        # Реализация выбора факультета!!!

        if (predicted_class == 'schedule'):
            surnames, group_num = extract_entities(user_text)
            #print(surnames, group_num)
            if len(surnames) == 0 and len(group_num) >= 1:
                group_num_current = group_num[0]
                faculties = parse_faculties()
                await message.answer("Выберите факультет:", reply_markup=create_faculties_keyboard(faculties))
                url = selected_faculty_link
                print(url, 'ССЫЛКА!!')
                link = get_group_schedule_link(str(group_num_current), url)
                if not(link is None):
                    inline_keyboard = InlineKeyboardMarkup()
                    button = InlineKeyboardButton("Перейти по ссылке", url=link)
                    inline_keyboard.add(button)
                    await message.answer(f"Расписание для {group_num_current} группы:", reply_markup=inline_keyboard)
                else:
                    inline_keyboard = InlineKeyboardMarkup()
                    button = InlineKeyboardButton("Перейти по ссылке", url=url)
                    inline_keyboard.add(button)
                    await message.answer("Расписание для этой группы не найдено\nСтраница с расписаниями для факультетов", reply_markup=inline_keyboard)

        # Создаем клавиатуру с ссылками, если ссылки не 'moke'
        links = class_answers[predicted_class]["links"]
        if links != ['moke']:
            inline_keyboard = InlineKeyboardMarkup()
            for link in links:
                button = InlineKeyboardButton("Перейти по ссылке", url=link)
                inline_keyboard.add(button)
            await message.reply(response_text, reply_markup=inline_keyboard)
        else:
            await message.reply(response_text)
        
        await bot.send_message(message.chat.id, "Это то что вы хотели узнать?", reply_markup=get_confirm_keyboard())
        
        await Form.confirm_response.set()
        await state.update_data(attempt=attempt, top_indices=top_indices, probabilities=probabilities)
    else:
        broken_response = class_answers['uknow']['text']
        await message.reply(broken_response)

# Обработчик нажатия на кнопку факультета или навигации
@dp.callback_query_handler(lambda c: c.data.startswith('faculty_') or c.data.startswith('prev_') or c.data.startswith('next_'))
async def process_callback(callback_query: types.CallbackQuery):
    faculties = parse_faculties()
    data = callback_query.data

    if data.startswith('faculty_'):
        selected_faculty = data.split('_', 1)[1]
        faculty_link = faculties[selected_faculty]

        # Сохранение ссылки на выбранный факультет в переменную
        global selected_faculty_link
        selected_faculty_link = faculty_link
        print(selected_faculty_link, "ВЫБОР В КЛАВЕ")
        await bot.answer_callback_query(callback_query.id)
        #await bot.answer_callback_query(callback_query.id)
        await bot.send_message(callback_query.from_user.id, f"Вы выбрали {selected_faculty}")
    else:
        if data.startswith('prev_'):
            page = int(data.split('_', 1)[1])
        elif data.startswith('next_'):
            page = int(data.split('_', 1)[1])
        
        await bot.answer_callback_query(callback_query.id)
        await bot.edit_message_text("Выберите факультет:", callback_query.from_user.id, callback_query.message.message_id, reply_markup=create_faculties_keyboard(faculties, page=page))

@dp.message_handler(lambda message: message.text in ["Да", "Нет"], state=Form.confirm_response)
async def handle_confirmation(message: types.Message, state: FSMContext):
    user_response = message.text
    user_data = await state.get_data()
    attempt = user_data["attempt"]
    top_indices = user_data["top_indices"]
    probabilities = user_data["probabilities"]

    if user_response == "Да":
        await message.reply("Спасибо за вопрос, я рад, что был вам полезен.", reply_markup=start_keyboard)
        await state.finish()
    elif user_response == "Нет":
        attempt += 1
        await send_response(message, state, top_indices, probabilities, attempt)
        await state.update_data(attempt=attempt)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
# import logging
# from aiogram import Bot, Dispatcher, types
# from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
# from aiogram.utils import executor
# from aiogram.contrib.middlewares.logging import LoggingMiddleware
# from aiogram.contrib.fsm_storage.memory import MemoryStorage
# from aiogram.dispatcher import FSMContext
# from aiogram.dispatcher.filters.state import State, StatesGroup
# import nn_cllassifier
# from nn_cllassifier import preprocess_text, load_own_model, predict
# import model_path
# import data_ready
# from data_ready import class_answers, LabelDict
# import numpy as np
# import ner_recogn
# from ner_recogn import extract_entities
# import requests
# from bs4 import BeautifulSoup

# API_TOKEN = '6501382603:AAGYDgjoV718Z_1p7PSViCRI8H4zW7RnyB0'

# logging.basicConfig(level=logging.INFO)

# storage = MemoryStorage()
# bot = Bot(token=API_TOKEN)
# dp = Dispatcher(bot, storage=storage)
# dp.middleware.setup(LoggingMiddleware())

# # для единоразовой загрузки модели при старте бота:
# tokenizer = None
# model = None
# # флаг что модель была успешно загружена
# load_model_success = False

# # Определение клавиатуры с кнопкой /start
# start_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
# start_button = types.KeyboardButton('/start')
# start_keyboard.add(start_button)

# # Клавиатура с ответами "Да" и "Нет"
# def get_confirm_keyboard():
#     confirm_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
#     yes_button = types.KeyboardButton('Да')
#     no_button = types.KeyboardButton('Нет')
#     confirm_keyboard.add(yes_button, no_button)
#     return confirm_keyboard

# class Form(StatesGroup):
#     confirm_response = State()
#     faculty_choice = State()

# @dp.message_handler(commands=['start'])
# async def send_welcome(message: types.Message):
#     print("bot started")
#     global tokenizer
#     global model
#     global load_model_success
#     # Загружаем модель и токенизатор один раз
#     if tokenizer is None or model is None:
#         model_path_dir = model_path.model_path
#         tokenizer, model = load_own_model(model_path_dir)
#         model.eval()  # Перевод модели в режим оценки
#         if not (tokenizer is None) and not (model is None):
#             load_model_success = True
#             print("model loaded")
    
#     # Приветственное сообщение
#     await message.reply("Привет! Я СГУшка: Бот-помощник. Отправь мне сообщение с вопросом, и я постараюсь помочь.", reply_markup=start_keyboard)

# @dp.message_handler(lambda message: message.text not in ["Да", "Нет"], state='*')
# async def handle_message(message: types.Message, state: FSMContext):
#     if not load_model_success:
#         await message.reply("Модель не загружена. Попробуйте позже.")
#         return
    
#     user_text = message.text
#     probabilities = predict(user_text, tokenizer, model)

#     if np.max(probabilities) <= 1e-3:
#         broken_response = class_answers['uknow']['text']
#         await message.reply(broken_response)
#     else:
#         top_indices = np.argsort(probabilities[0])[::-1]
#         current_attempt = 0

#         await send_response(message, state, top_indices, probabilities[0], current_attempt)

# async def send_response(message, state: FSMContext, top_indices, probabilities, attempt):
#     logging.info(f"Attempt: {attempt}")
#     logging.info(f"Top indices: {top_indices}")
#     logging.info(f"Probabilities: {probabilities}")

#     if attempt < min(2, len(top_indices)) and probabilities[top_indices[attempt]] >= 1e-4:
#         predicted_class_index = top_indices[attempt]
#         predicted_class = LabelDict[predicted_class_index]
#         response_text = class_answers[predicted_class]["text"]

#         if predicted_class == 'schedule':
#             surnames, group_numbers = extract_entities(message.text)
#             if surnames:
#                 teacher_name = surnames[0]
#                 schedule_links = get_teacher_schedule_links(teacher_name)
#                 if schedule_links:
#                     inline_keyboard = InlineKeyboardMarkup()
#                     if 'class_schedule' in schedule_links:
#                         button = InlineKeyboardButton("Расписание занятий", url=schedule_links['class_schedule'])
#                         inline_keyboard.add(button)
#                     if 'exam_schedule' in schedule_links:
#                         button = InlineKeyboardButton("Расписание сессии", url=schedule_links['exam_schedule'])
#                         inline_keyboard.add(button)
#                     await message.reply(f"Расписание преподавателя {teacher_name}:", reply_markup=inline_keyboard)
#                     await state.finish()
#                     return
#                 else:
#                     await message.reply(f"Не удалось найти расписание для преподавателя {teacher_name}.")
#             elif group_numbers:
#                 group_number = group_numbers[0]
#                 faculties = get_faculties()
#                 if faculties:
#                     inline_keyboard = InlineKeyboardMarkup()
#                     for faculty in faculties:
#                         button = InlineKeyboardButton(faculty, callback_data=f"faculty_{faculty}_{group_number}")
#                         inline_keyboard.add(button)
#                     await message.reply("Выберите факультет для поиска расписания группы:", reply_markup=inline_keyboard)
#                     await Form.faculty_choice.set()
#                     return
#                 else:
#                     await message.reply("Не удалось загрузить список факультетов.")
#             else:
#                 await message.reply("Не удалось распознать фамилию преподавателя или номер группы.")
#         else:
#             links = class_answers[predicted_class]["links"]
#             if links != ['moke']:
#                 inline_keyboard = InlineKeyboardMarkup()
#                 for link in links:
#                     button = InlineKeyboardButton("Перейти по ссылке", url=link)
#                     inline_keyboard.add(button)
#                 await message.reply(response_text, reply_markup=inline_keyboard)
#             else:
#                 await message.reply(response_text)
        
#         await bot.send_message(message.chat.id, "Это то что вы хотели узнать?", reply_markup=get_confirm_keyboard())
#         await Form.confirm_response.set()
#         await state.update_data(attempt=attempt, top_indices=top_indices, probabilities=probabilities)
#     else:
#         broken_response = class_answers['uknow']['text']
#         await message.reply(broken_response)

# @dp.message_handler(lambda message: message.text in ["Да", "Нет"], state=Form.confirm_response)
# async def handle_confirmation(message: types.Message, state: FSMContext):
#     user_response = message.text
#     user_data = await state.get_data()
#     attempt = user_data["attempt"]
#     top_indices = user_data["top_indices"]
#     probabilities = user_data["probabilities"]

#     if user_response == "Да":
#         await message.reply("Спасибо за вопрос, я рад, что был вам полезен.", reply_markup=start_keyboard)
#         await state.finish()
#     elif user_response == "Нет":
#         attempt += 1
#         await send_response(message, state, top_indices, probabilities, attempt)
#         await state.update_data(attempt=attempt)

# @dp.callback_query_handler(lambda c: c.data.startswith('faculty_'), state=Form.faculty_choice)
# async def handle_faculty_choice(callback_query: types.CallbackQuery, state: FSMContext):
#     data = callback_query.data.split('_')
#     faculty = data[1]
#     group_number = data[2]
#     url = f"https://www.sgu.ru/schedule/{faculty}"
#     group_schedule_link = get_group_schedule_link(group_number, url)
#     if group_schedule_link:
#         await bot.send_message(callback_query.from_user.id, f"Ссылка на расписание группы {group_number}:", reply_markup=InlineKeyboardMarkup().add(InlineKeyboardButton("Перейти по ссылке", url=group_schedule_link)))
#     else:
#         await bot.send_message(callback_query.from_user.id, f"Не удалось найти расписание для группы {group_number} на факультете {faculty}.")
#     await state.finish()

# def get_faculties():
#     url = "https://www.sgu.ru/schedule"
#     response = requests.get(url)
#     if response.status_code != 200:
#         print(f"Ошибка доступа к странице. Код состояния: {response.status_code}")
#         return None

#     soup = BeautifulSoup(response.content, 'html.parser')
#     faculty_links = soup.find_all('a', href=True)
#     faculties = [link.text for link in faculty_links if 'schedule' in link['href']]
#     return faculties

# if __name__ == '__main__':
#     executor.start_polling(dp, skip_updates=True)
