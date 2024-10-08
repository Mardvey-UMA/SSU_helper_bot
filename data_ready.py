class_answers = {
    "schedule": {
        "text": "Для того чтобы узнать необходимое Вам расписание необходимо перейти по ссылке и найти нужную Вам группу. В случае возникновения накладок, обратитесь к диспетчеру, который находится в деканате вашего факультета.",
        "links": ["https://www.sgu.ru/schedule"]
    },
    "pay": {
        "text": "Всю информацию об оплате обучения, включая реквизиты, порядок оплаты и приказы со стоимостью Вы можете найти по следующей ссылке: ",
        "links": ["https://www.sgu.ru/sveden/paid_edu"]
    },
    "events": {
        "text": "Новости о любых мероприятиях, проходящих в университете Вы можете узнать на главной странице университета, а также в группа в ВК и Telegram.",
        "links": ["https://www.sgu.ru/"]
    },
    "military": {
        "text": "В СГУ есть возможность прохождения военной кафедры на базе СГТУ, для получения всей дополнительной информации перейдите по ссылке.",
        "links": ["https://www.sgu.ru/voennoe-obuchenie"]
    },
    "statement_general": {
        "text": "Для получения справки о том, что вы являетесь студентом и/или справки в военкомат необходимо заполнить форму в личном кабинете Ipsilon. Для получения справки-вызов на работу необходимо прислать справку с места работы на адрес kniit@info.sgu.ru, указав ФИО и номер группы.",
        "links": ["https://ipsilon.sgu.ru/online_dekanat"]
    },
    "holidays": {
        "text": "Вся информация о графике учебного процесса, времени сессии и каникул находится здесь.",
        "links": ["https://www.old.sgu.ru/structure/edudep/grafiki-uchebnogo-processa"]
    },
    "finassist": {
        "text": "Всю информацию о стипендии, а также, что делать если она не пришла, пришла в неправильном размере и тд можно узнать в Плановом финансовом управлении по номеру: 52-30-32.",
        "links": ["moke"]
    },
    "dorm": {
        "text": "Всю информацию о возможности проживания в общежитии, необходимых документах и тд, Вы можете узнать перейдя по ссылке.",
        "links": ["https://www.sgu.ru/studencheskie-obschezhitiya"]
    },
    "loc": {
        "text": "Информация о расположении корпусов.",
        "links": ["https://www.sgu.ru/karta-kampusa"]
    },
    "grade": {
        "text" : 'В случае получения оценки "неудовлетворительно", либо неявки на экзамен, студент может пересдать дисциплину в сроки отведенные для пересдач. Расписание пересдач можно найти по ссылке:',
        "links" : ["https://course.sgu.ru/course/view.php?id=1491"]
    },
    "uknow":{
        "text": "Извините ваш вопрос не понятен, попробуйте сформулировать его по другому и отправить еще раз.",
        "links": ["moke"]
    }
}
# Определение словарей меток
LabelDict = {
    0: 'statement_general',
    1: 'dorm',
    2: 'events',
    3: 'finassist',
    4: 'grade',
    5: 'loc',
    6: 'military',
    7: 'holidays',
    8: 'pay',
    9: 'schedule',
}
# инвертное
LabelDictInv = {
    'statement_general' : 0,
    'dorm' : 1,
    'events' : 2,
    'finassist' : 3,
    'grade' : 4,
    'loc' : 5,
    'military' : 6,
    'holidays' : 7,
    'pay' : 8,
    'schedule' : 9
}
facult_name = {
    'Биологический факультет' : 'Биологический',
    'Географический факультет' : 'Географический',
    'Геологический факультет' : 'Геологический',
    'Институт дополнительного профессионального образования' : 'ИДПО',
    'Институт искусств' : 'ИИ',
    'Институт истории и международных отношений' : 'ИИиМО',
    'Институт физики' : 'ИнФиз',
    'Институт физической культуры и спорта' : 'ИФКиС',
    'Институт филологии и журналистики' : 'ИФиЖ',
    'Институт химии': 'ИнХим',
    'Механико-математический факультет': 'МехМат',
    'Социологический факультет': 'Социологический',
    'Факультет иностранных языков и лингводидактики': 'ФИЯиЛ',
    'Факультет компьютерных наук и информационных технологий': 'КНиИТ',
    'Факультет психологии': 'Психологический',
    'Факультет психолого-педагогического и специального образования': 'ППиСО',
    'Факультет фундаментальной медицины и медицинских технологий': 'ФМиМТ',
    'Философский факультет': 'Филосовский',
    'Экономический факультет': 'Экономический',
    'Юридический факультет': 'Юридический',
    'Геологический колледж': 'Гео колледж',
    'Колледж радиоэлектроники им. П.Н. Яблочкова': 'Колледж Яблочкова',
    'Психолого-педагогический факультет': 'Психолого-педагогический',
    'Факультет математики и естественных наук': 'ФМиЕН',
    'Филологический факультет': 'Филологический'
}
facult_name_inv = {
    'Биологический':'Биологический факультет' ,
    'Географический':'Географический факультет', 
    'Геологический':'Геологический факультет' ,
    'ИДПО':'Институт дополнительного профессионального образования' ,
    'ИИ':'Институт искусств' ,
    'ИИиМО':'Институт истории и международных отношений' ,
    'ИнФиз':'Институт физики' ,
    'ИФКиС':'Институт физической культуры и спорта' ,
    'ИФиЖ':'Институт филологии и журналистики' ,
    'ИнХим':'Институт химии',
    'МехМат':'Механико-математический факультет',
    'Социологический':'Социологический факультет',
    'ФИЯиЛ':'Факультет иностранных языков и лингводидактики',
    'КНиИТ':'Факультет компьютерных наук и информационных технологий',
    'Психологический':'Факультет психологии',
    'ППиСО':'Факультет психолого-педагогического и специального образования',
    'ФМиМТ':'Факультет фундаментальной медицины и медицинских технологий',
    'Филосовский':'Философский факультет',
    'Экономический':'Экономический факультет',
    'Юридический':'Юридический факультет',
    'Гео колледж':'Геологический колледж',
    'Колледж Яблочкова':'Колледж радиоэлектроники им. П.Н. Яблочкова',
    'Психолого-педагогический':'Психолого-педагогический факультет',
    'ФМиЕН':'Факультет математики и естественных наук',
    'Филологический' :'Филологический факультет'
}