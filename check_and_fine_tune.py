import sqlite3
import os

def check_new_data():
    conn = sqlite3.connect('/home/dev-bot/ssu_project/scripts/questions.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT COUNT(*) FROM questions WHERE used = 0 GROUP BY label HAVING COUNT(*) >= 10
    ''')
    result = cursor.fetchall()
    conn.close()
    return len(result) > 0

if check_new_data():
    os.system("/home/dev-bot/ssu_project/venv/bin/python /home/dev-bot/ssu_project/scripts/fine_tuning.py")
    os.system("/home/dev-bot/ssu_project/venv/bin/python /home/dev-bot/ssu_project/scripts/test_model.py")
else:
    print("Недостаточно новых примеров")
