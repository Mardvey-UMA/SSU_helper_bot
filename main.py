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
import conf
from conf import TOKEN
import database

logging.basicConfig(level=logging.INFO)

storage = MemoryStorage()
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=storage)
dp.middleware.setup(LoggingMiddleware())
selected_faculty_link = None

# для единоразовой загрузки модели при старте бота:
tokenizer = None
model = None
# флаг что модель была успешно загружена
load_model_success = False

# Определение клавиатуры с кнопками основного меню
main_menu_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
question_button = types.KeyboardButton('❓ Задать вопрос', callback_data='/start')
instruction_button = types.KeyboardButton('📋 Инструкция использования')
faq_button = types.KeyboardButton('❔ Частые вопросы')
report_button = types.KeyboardButton('🚨 Сообщить об ошибке')
main_menu_keyboard.add(question_button, instruction_button, faq_button, report_button)

def parse_faculties():
    url = "https://old.sgu.ru/schedule"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    faculties = {}
    for link in soup.select('a[href*="/schedule/"]'):
        faculty_name = ' '.join([word for word in link.text.strip().split()])
        faculty_link = "https://old.sgu.ru" + link['href']
        faculties[faculty_name] = faculty_link
    
    return faculties

def create_faculties_keyboard(faculties, page=0, items_per_page=4):
    keyboard = InlineKeyboardMarkup()
    faculties_list = list(faculties.items())
    start = page * items_per_page
    end = start + items_per_page
    for faculty_name, faculty_link in faculties_list[start:end]:
        keyboard.add(InlineKeyboardButton(text=data_ready.facult_name[faculty_name], callback_data=f"faculty_{data_ready.facult_name[faculty_name]}"))
        
    navigation_buttons = []
    if start > 0:
        navigation_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"prev_{page-1}"))
    if end < len(faculties):
        navigation_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"next_{page+1}"))
    
    if navigation_buttons:
        keyboard.row(*navigation_buttons)
    
    return keyboard

# Клавиатура с ответами "Да" и "Нет"
def get_confirm_keyboard():
    confirm_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    yes_button = types.KeyboardButton('✔️ Да')
    no_button = types.KeyboardButton('❌ Нет')
    confirm_keyboard.add(yes_button, no_button)
    return confirm_keyboard

class Form(StatesGroup):
    confirm_response = State()
    report_issue = State()

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
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
    
    await message.reply("Привет! Я СГУшка: Бот-помощник. Отправь мне сообщение с вопросом, и я постараюсь помочь.", reply_markup=main_menu_keyboard)

@dp.message_handler(state=Form.report_issue, content_types=['text', 'photo'])
async def handle_report_issue(message: types.Message, state: FSMContext):
    if message.text and message.text.lower() == 'отмена':
        await message.reply("Отправка сообщения об ошибке отменена.", reply_markup=main_menu_keyboard)
        await state.finish()
        return

    issue_text = message.caption if message.photo else message.text
    photos = message.photo

    if photos:
        for photo in photos:
            photo_id = photo.file_id
            await bot.send_photo(chat_id='5946671450', photo=photo_id, caption=f"Сообщение об ошибке от {message.from_user.id}: {issue_text}")
    else:
        await bot.send_message(chat_id='5946671450', text=f"Сообщение об ошибке от {message.from_user.id}: {issue_text}")
    
    await message.reply("Спасибо за ваше сообщение. Мы постараемся исправить ошибку в ближайшее время.", reply_markup=main_menu_keyboard)
    await state.finish()

@dp.message_handler(lambda message: message.text not in ["✔️ Да", "❌ Нет"], state='*')
async def handle_message(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state == Form.report_issue.state:
        await handle_report_issue(message, state)
        return

    if len(message.text) > 300:
        await message.reply("Извините, слишком длинное сообщение, попробуйте сократить вопрос.", reply_markup=main_menu_keyboard)
        return

    if not load_model_success:
        await message.reply("Модель не загружена. Используйте /start для начала работы с ботом.")
        return
    
    user_text = message.text
    if user_text == '/start' or user_text == '❓ Задать вопрос':
        await send_welcome(message)
        return
    elif user_text == '📋 Инструкция использования':
        await message.reply("[Инструкция](https://telegra.ph/Instrukciya-ispolzovaniya-bota-06-06)", parse_mode="markdown")
        return
    elif user_text == '❔ Частые вопросы':
        await message.reply("""Список частых вопросов:
                            \n1. Как получить справку, что я студент?
                            \n2. Как заказать справку-вызов?
                            \n3. Какие документы нужны для общежития?
                            \n4. Сколько стоит обучение?
                            \n5. Где взять реквизиты для оплаты?
                            \n6. Когда нужно оплатить обучение?""")
        return
    elif user_text == '🚨 Сообщить об ошибке':
        cancel_button = KeyboardButton('Отмена')
        cancel_keyboard = ReplyKeyboardMarkup(resize_keyboard=True).add(cancel_button)
        await message.reply("Расскажите об ошибке или отправьте скриншот с проблемой и описанием:", reply_markup=cancel_keyboard)
        await Form.report_issue.set()
        return
    
    probabilities = predict(user_text, tokenizer, model)

    if np.max(probabilities) <= 1e-4:
        database.add_unclassified_question(message.from_user.id, message.text)
        broken_response = class_answers['uknow']['text']
        await message.reply(broken_response, reply_markup=main_menu_keyboard)
    else:
        top_indices = np.argsort(probabilities[0])[::-1]
        current_attempt = 0

        await state.update_data(original_text=user_text)  # Сохранение оригинального текста вопроса
        await send_response(message, state, top_indices, probabilities[0], current_attempt)

async def send_response(message: types.Message, state: FSMContext, top_indices, probabilities, attempt):
    logging.info(f"Attempt: {attempt}")
    logging.info(f"Top indices: {top_indices}")
    logging.info(f"Probabilities: {probabilities}")
    
    if attempt < min(2, len(top_indices)) and probabilities[top_indices[attempt]] >= 1e-3:
        predicted_class_index = top_indices[attempt]
        predicted_class = LabelDict[predicted_class_index]
        response_text = class_answers[predicted_class]["text"]

        if predicted_class == 'schedule':
            user_data = await state.get_data()
            original_text = user_data['original_text']  # Получение оригинального текста вопроса
            surnames, group_num = extract_entities(original_text)
            if len(surnames) == 0 and len(group_num) >= 1:
                group_num_current = group_num[0]
                faculties = parse_faculties()
                await message.answer("Выберите факультет:", reply_markup=create_faculties_keyboard(faculties))
                await state.update_data(group_num=group_num_current)
                await state.update_data(predicted_class=predicted_class)
                return
            elif len(surnames) >= 1:
                surname_current = surnames[0]
                schedule_links = get_teacher_schedule_links(surname_current)
                if schedule_links:
                    inline_keyboard = InlineKeyboardMarkup()
                    if schedule_links.get('class_schedule'):
                        button_class_schedule = InlineKeyboardButton("Расписание занятий", url=schedule_links.get('class_schedule'))
                        inline_keyboard.add(button_class_schedule)
                    if schedule_links.get('exam_schedule'):
                        button_exam_schedule = InlineKeyboardButton("Расписание сессии", url=schedule_links.get('exam_schedule'))
                        inline_keyboard.add(button_exam_schedule)
                    await message.reply(f"Расписание для преподавателя {surname_current}:", reply_markup=inline_keyboard)
                    return
                else:
                    await state.update_data(surname=surname_current)
                    await state.update_data(predicted_class=predicted_class)

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
        database.add_unclassified_question(message.from_user.id, original_text)
        broken_response = class_answers['uknow']['text']
        await message.reply(broken_response, reply_markup=main_menu_keyboard)

@dp.callback_query_handler(lambda c: c.data.startswith('faculty_') or c.data.startswith('prev_') or c.data.startswith('next_'))
async def process_callback(callback_query: types.CallbackQuery, state: FSMContext):
    faculties = parse_faculties()
    data = callback_query.data

    if data.startswith('faculty_'):
        selected_faculty = data.split('_', 1)[1]
        faculty_link = faculties[data_ready.facult_name_inv[selected_faculty]]
        global selected_faculty_link
        selected_faculty_link = faculty_link
        await state.update_data(selected_faculty_link=faculty_link)
        await bot.answer_callback_query(callback_query.id)
        await bot.send_message(callback_query.from_user.id, f"Вы выбрали {data_ready.facult_name_inv[selected_faculty]}")

        user_data = await state.get_data()
        group_num_current = user_data.get('group_num')

        if group_num_current:
            link = get_group_schedule_link(str(group_num_current), selected_faculty_link)
            if link:
                inline_keyboard = InlineKeyboardMarkup()
                button = InlineKeyboardButton("Перейти по ссылке", url=link)
                inline_keyboard.add(button)
                await bot.send_message(callback_query.from_user.id, f"Расписание для {group_num_current} группы:", reply_markup=inline_keyboard)
            else:
                inline_keyboard = InlineKeyboardMarkup()
                button = InlineKeyboardButton("Перейти по ссылке", url=selected_faculty_link)
                inline_keyboard.add(button)
                await bot.send_message(callback_query.from_user.id, "Расписание для этой группы не найдено\nСтраница с расписаниями для факультетов", reply_markup=inline_keyboard)
            await state.finish()
    else:
        if data.startswith('prev_'):
            page = int(data.split('_', 1)[1])
        elif data.startswith('next_'):
            page = int(data.split('_', 1)[1])
        
        await bot.answer_callback_query(callback_query.id)
        await bot.edit_message_text("Выберите факультет:", callback_query.from_user.id, callback_query.message.message_id, reply_markup=create_faculties_keyboard(faculties, page=page))

@dp.message_handler(lambda message: message.text in ["✔️ Да", "❌ Нет"], state=Form.confirm_response)
async def handle_confirmation(message: types.Message, state: FSMContext):
    user_response = message.text
    user_data = await state.get_data()
    attempt = user_data["attempt"]
    top_indices = user_data["top_indices"]
    probabilities = user_data["probabilities"]
    original_text = user_data["original_text"] 

    if user_response == "✔️ Да":
        if attempt > 0:
            predicted_class = LabelDict[top_indices[attempt]]
            database.add_question(message.from_user.id, original_text, int(top_indices[attempt]))
        await message.reply("Спасибо за вопрос, я рад, что был вам полезен.", reply_markup=main_menu_keyboard)
        await state.finish()
    elif user_response == "❌ Нет":
        attempt += 1
        if attempt < 2 and probabilities[top_indices[attempt]] >= 1e-4:
            await send_response(message, state, top_indices, probabilities, attempt)
            await state.update_data(attempt=attempt)
        else:
            database.add_unclassified_question(message.from_user.id, original_text)
            await message.reply("Извините, я не смог понять ваш вопрос. Он был отправлен на проверку.", reply_markup=main_menu_keyboard)
            await state.finish()

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
