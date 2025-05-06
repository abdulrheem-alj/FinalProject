import os
from flask import Flask, Response, jsonify, render_template, request

from methods import runSmartAgent
import speech_recognition as sr
from flask_cors import CORS
from werkzeug.utils import secure_filename


app = Flask(__name__)
CORS(app) 
 

 
# مسار المجلد الذي ستُحفظ فيه الصور
app.config['UPLOAD_FOLDER'] = 'input'
# تحدد الصيغ المسموح بها
ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg'}
ALLOWED_AUDIO_EXT = {'wav', 'webm'}




@app.route('/')
def home():
    with open("templates/index.html", encoding="utf-8") as f:
        html = f.read()
    return Response(html, mimetype='text/html')

def  multiple_values(result ):
      
    return isinstance(result, (tuple, list, dict)) 


@app.route('/SmartAgent/<user_input>', methods=['GEt'])
def SmartAgent(user_input):
    
    result  , lang , sourse = runSmartAgent(user_input) 
    if multiple_values (result):
        result = result["answer"] 


    if lang == 'ar' : idLang = 1 
    else: idLang = 2 

    return  jsonify({
            'success': True, 
            "result": result ,
            'lang': idLang ,
            'sourse': sourse,
    })
 


def allowed_file(filename, allowed_exts):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in allowed_exts
    )

@app.route('/upload', methods=['POST'])
def upload_media():
    # إذا أرسل المستخدم صورة
    if 'image' in request.files:
        file = request.files['image']
        exts = ALLOWED_IMAGE_EXT
        key = 'image'
    # أو إذا أرسل ملف صوت
    elif 'audio' in request.files:
        file = request.files['audio']
        exts = ALLOWED_AUDIO_EXT
        key = 'audio'
    else:
        return jsonify({'error': 'لم يتم إرسال صورة أو ملف صوت'}), 400

    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400

    if allowed_file(file.filename, exts):

        if(key == 'image'):
            filename ="img.png"
        elif(key == 'audio'):
            filename ="audio.wav"
        else:
            filename = secure_filename(file.filename)
       
       
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        result ,lang , sourse = runSmartAgent("input/" + filename) 
      
            
        if lang == 'ar' : idLang = 1 
        else: idLang = 2 

        return  jsonify({
            'success': True, 
            'filename': filename,
            "result": result ,
            'type': key, 
            'lang': idLang ,
            'sourse': sourse,
        }), 200
 

    return jsonify({'error': f'صيغة {key} غير مدعومة'}), 400
 

if __name__ == '__main__':
    app.run(debug=True)



