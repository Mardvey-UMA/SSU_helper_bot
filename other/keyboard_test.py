import requests
from bs4 import BeautifulSoup
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils import executor
import logging

API_TOKEN = '6501382603:AAGYDgjoV718Z_1p7PSViCRI8H4zW7RnyB0'

# Включаем логирование
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

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

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    faculties = parse_faculties()
    await message.answer("Выберите факультет:", reply_markup=create_faculties_keyboard(faculties))

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
        print(selected_faculty_link)
        await bot.answer_callback_query(callback_query.id)
        await bot.send_message(callback_query.from_user.id, f"Вы выбрали {selected_faculty}. Ссылка на расписание: {faculty_link}")
    else:
        if data.startswith('prev_'):
            page = int(data.split('_', 1)[1])
        elif data.startswith('next_'):
            page = int(data.split('_', 1)[1])
        
        await bot.answer_callback_query(callback_query.id)
        await bot.edit_message_text("Выберите факультет:", callback_query.from_user.id, callback_query.message.message_id, reply_markup=create_faculties_keyboard(faculties, page=page))

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)