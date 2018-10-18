import speech_recognition as sr
from os import path

r = sr.Recognizer()
IBM_USERNAME = "a87baa11-b3c6-471f-b875-a0dcbc4254b5"
IBM_PASSWORD = "Xx0rZwBbXQlY"
#output = open('/Users/Vicky/Desktop/dist/text.txt','a+')
with open('lab/ComParE2016_Deception.tsv') as f:
    for line in f.readlines()[1:]:
        lists = line.split(',')
        AUDIO_FILE = 'wav/' + lists[0]
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)
            try:
                text = lists[0]+','+r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD)
                print(text)
            #    print>>output,text
            except sr.UnknownValueError:
                print("Could not understand audio")
             #   print>>output,lists[0]
            except sr.RequestError as e:
                print("Could not request results from IBM Speech to Text service; {0}".format(e))
             #   print>>output,lists[0]
output.close()
f.close()
