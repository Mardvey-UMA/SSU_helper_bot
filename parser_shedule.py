import requests
from bs4 import BeautifulSoup

def get_group_schedule_link(group_number, url):
    #url = "https://www.sgu.ru/schedule/knt"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Ошибка доступа к странице. Код состояния: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    links = soup.find_all('a', href=True)

    for link in links:
        if group_number in link.text:
            return f"https://www.sgu.ru{link['href']}"
    
    #https://www.sgu.ru
    print(f"Ссылка для группы {group_number} не найдена.")
    return None

def get_teacher_page_url(teacher_name):
    base_url = "https://www.sgu.ru/person"
    response = requests.get(base_url)

    if response.status_code != 200:
        print(f"Не удалось получить доступ к списку преподавателей. Код состояния: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    teacher_links = soup.find_all('a', href=True)

    for link in teacher_links:
        if teacher_name.lower() in link.text.lower():
            return f"https://www.sgu.ru{link['href']}"

    print(f"Преподаватель с фамилией '{teacher_name}' не найден.")
    return None

def get_teacher_schedule_links(teacher_name):
    teacher_url = get_teacher_page_url(teacher_name)
    
    if not teacher_url:
        return None

    response = requests.get(teacher_url)
    
    if response.status_code != 200:
        print(f"Не удалось получить доступ к странице преподавателя. Код состояния: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    schedule_links = {}
    
    social_section = soup.find('div', class_='social')
    
    if not social_section:
        print(f"У преподавателя {teacher_name} нет информации о расписании.")
        return None

    links = social_section.find_all('a', href=True)
    for link in links:
        if "расписание занятий" in link.text.lower():
            schedule_links['class_schedule'] = "https://www.sgu.ru" + link['href']
        elif "расписание сессии" in link.text.lower():
            schedule_links['exam_schedule'] = "https://www.sgu.ru" + link['href']

    if not schedule_links:
        print(f"У преподавателя {teacher_name} нет информации о расписании.")
        return None
    #print(f"Ссылка на расписание занятий: {schedule_links.get('class_schedule')}")
    #print(f"Ссылка на расписание сессии: {schedule_links.get('exam_schedule')}")
    return schedule_links


