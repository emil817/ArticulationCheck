import telebot
import os
import tensorflow as tf
from catboost import CatBoostClassifier

from Tokens import telebot_token
from Analyzers import analyse_video, analyse_audio

if not os.path.exists("content/video"):
    os.makedirs("content/video")
    os.makedirs("content/audio")

numf = 0
bot = telebot.TeleBot(telebot_token)

audio_model = tf.keras.models.load_model('audio_model2')
video_model = CatBoostClassifier()
video_model.load_model("video_model")

@bot.message_handler(commands=["start"])
def start(m, res=False):
    bot.send_message(m.chat.id, 'Я бот для оценки правильности произношения\nМогу определить насколько правильно вы произносите звуки\nОтправьте мне видео, где вы что-нибудь говорите')

@bot.message_handler (content_types = ['text'])
def Text (Message):
    bot.send_message(Message.chat.id, 'Здравствуйте, ' + Message.from_user.first_name + '\nМогу определить насколько правильно вы произносите звуки, для этого отпраьте мне видео, где вы что-нибудь говорите')

@bot.message_handler (content_types = ['video'])
def Send_Text (Message):
    bot.send_message(Message.chat.id, 'Обработка видео...')
    File_info = bot.get_file(Message.video.file_id )
    downloaded_file = bot.download_file (File_info.file_path)
    with open(os.getcwd() + "/content/video/" + str(numf)+ ".mp4", 'wb') as new_file:
        new_file.write(downloaded_file)
    video_accuracy = analyse_video(os.getcwd() + "/content/video/" + str(numf)+ ".mp4", video_model)
    audio_accuracy = analyse_audio(os.getcwd() + "/content/video/" + str(numf)+ ".mp4", numf, audio_model)

    os.remove(os.getcwd() + "/content/video/" + str(numf)+ ".mp4")
    acc = (audio_accuracy + video_accuracy)/2
    bot.reply_to (Message, Message.from_user.first_name + ",\nВаше видео проанализировано,\nВы говорите {:.2f}% звуков правильно".format(acc))
    if acc < 100:
        bot.send_message(Message.chat.id, "Вы что-то не выговариваете!")
    else:
        bot.send_message(Message.chat.id, "Вы говорите всё звуки правильно!")


@bot.message_handler(content_types=['sticker'])
def voice_processing(message):
    bot.send_message(message.chat.id, "Извините, но я не понимаю стикеры")
    bot.send_sticker(message.chat.id, "CAACAgIAAxkBAAEK6IplcC9lKTwkcJAZGcX2g6oiMFzWDgACGAADwDZPE9b6J7-cahj4MwQ")



bot.polling(none_stop=True, interval=0)