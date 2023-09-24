from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pyrebase
from pymongo import MongoClient
import jinja2
#from dotenv import load_dotenv
#load_dotenv('.env')
import os
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import io


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY','1324456789')
client = MongoClient(os.getenv('mongostr'), serverSelectionTimeoutMS=60000)
db = client["Food"]
admins = []
admin_mails = []


db = client["Food"]
DB = client['HacksForU']

Admin = iter(db["admin"].find())
for records in Admin:
    admins.append(records['acnt_local_id'])
    admin_mails.append(records['mail_id'])

config={
  "apiKey":os.getenv("apiKey") ,
  "authDomain":os.getenv("authDomain") ,
  "databaseURL":os.getenv("databaseURL") ,
  "projectId": os.getenv("projectId"),
  "storageBucket": os.getenv("storageBucket"),
  "messagingSenderId": os.getenv("messagingSenderId"),
  "appId": os.getenv("appId"),
  "measurementId": os.getenv("measurementId")}

def unique_count(lst):
    return len(set(lst))

app.jinja_env.filters['unique_count'] = unique_count


# Initialize Firebase Admin SDK
firebase=pyrebase.initialize_app(config)
auth = firebase.auth()


# Load DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

db = client["Food"]  # Replace with your database name
collection = db["items"]  # Replace with your collection name

@app.route('/lost_and_found', methods=['GET', 'POST'])
def lost_and_found():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']

        # Check if an image file is uploaded
        if 'image' not in request.files:
            return redirect(request.url)

        image_file = request.files['image']

        # Check if the file is empty
        if image_file.filename == '':
            return redirect(request.url)

        # Check if the file is an image
        if image_file and allowed_file(image_file.filename):
            # Read and process the uploaded image
            image = Image.open(image_file.stream)
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # Process the detection results
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            # Prepare the results for rendering
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                detection_info = {
                    "label": model.config.id2label[label.item()],
                    "score": round(score.item(), 3),
                    "box": box
                }
                detections.append(detection_info)

                # Check if label exists in MongoDB
                existing_record = collection.find_one({"label": detection_info["label"]})

                if existing_record:
                    # If label exists, update the record
                    updated_record = {
                        "$push": {
                            "items": {
                                "name": name,
                                "phone": phone,
                                "images": [image_file.filename]
                            }
                        }
                    }
                    collection.update_one({"_id": existing_record["_id"]}, updated_record)
                else:
                    # If label doesn't exist, create a new record
                    new_record = {
                        "label": detection_info["label"],
                        "items": [
                            {
                                "name": name,
                                "phone": phone,
                                "images": [image_file.filename]
                            }
                        ]
                    }
                    collection.insert_one(new_record)

            return render_template('result.html', image=image, detections=detections)

    return render_template('lost_and_found.html')

@app.route('/list_labels')
def list_labels():
    labels = collection.distinct("label")
    return render_template('list_labels.html', labels=labels)

@app.route('/items/<label>')
def items_with_label(label):
    items = []
    records = collection.find({"label": label})
    for record in records:
        for item in record.get("items", []):
            items.append(item)
    return render_template('items_with_label.html', label=label, items=items)

# Helper function to check file extensions
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions



@app.route('/notices', methods=['GET', 'POST'])
def notices():
    NoticeBoard = db["Notices"]
    notices = iter(NoticeBoard.find())
    return render_template('noticeBoard.html', notices=notices)

# Home page
@app.route('/')
def index():
    return redirect('/login')

# Signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            user = auth.create_user_with_email_and_password(email, password)
            return redirect(url_for('login'))
        except Exception as e:
            error_message = str(e)
            return render_template('signup.html', error_message=error_message)
    return render_template('signup.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user_token'] = user['idToken']
            if user['localId'] in admins:
                return redirect(url_for('admin'))
            return redirect(url_for('dashboard'))
        except Exception as e:
            error_message = str(e)
            return render_template('login.html', error_message=error_message)
    return render_template('login.html')

# Dashboard page
@app.route('/dashboard')
def dashboard():
    user_token = session.get('user_token')
    if user_token:
        try:
            # Use the user_token to fetch user-specific data from Firebase Realtime Database
            user_email = auth.get_account_info(user_token)['users'][0]['email']
            return render_template('dashboard.html', user_email=user_email)
        except Exception as e:
            print("Error fetching user data:", e)
            return redirect(url_for('login'))
    return redirect(url_for('login'))


@app.route('/vote', methods=['GET', 'POST'])
def vote():
    Poll = db['Poll']
    polls = list(Poll.find())  # Convert polls to a list instead of an iterator
    if request.method == 'POST':
        selected_options = {}
        for poll in polls:
            poll_title = poll['title']
            selected_options[poll_title] = request.form.get(poll_title)
            Poll.update_one(
                {"title": poll_title},
                {"$addToSet": {f"{selected_options[poll_title]}": auth.get_account_info(session.get('user_token'))['users'][0]['email']}},
            )
            Poll.update_one({"title": poll_title}, {"$push": {"voters": auth.get_account_info(session.get('user_token'))['users'][0]['email']}})
        return redirect(url_for('vote'))
    return render_template('Polls.html', polls=polls)



# Logout
@app.route('/logout')
def logout():
    session.pop('user_token', None)
    return redirect(url_for('index'))

###############################################################################################

@app.route('/admin')
def admin():
    user_token = session.get('user_token')
    if user_token:
        if auth.get_account_info(user_token)['users'][0]['email'] in admin_mails:
            return render_template('admin.html')


@app.route('/create_poll', methods=['POST'])
def create_poll():
    Polls = db["Poll"]
    option1 = request.form.get('option1')
    option2 = request.form.get('option2')
    new_poll = {
        "title": f"{option1} vs {option2}",
        "options": [option1,option2],
        f"{option1}": [],
        f"{option2}": [],
        "voters": []
    }
    Polls.insert_one(new_poll)
    return redirect('/handle_polls')

@app.route('/remove_polls', methods=['POST'])
def remove_notices():
    Polls = db["Poll"]
    selected_polls = request.form.getlist('remove')
    for title in selected_polls:
        Polls.delete_one({"title": title})
    
    return redirect('/handle_polls')


@app.route('/handle_polls', methods=['GET'])
def handle_polls():
    Polls = db["Poll"]
    polls = list(Polls.find())
    return render_template('ChangePolls.html', polls=polls)


@app.route('/create_notice', methods=['POST'])
def create_notice():
    NoticeBoard = db["Notices"]
    category = request.form.get('category')
    notice = request.form.get('notice')
    NoticeBoard.update_one(
        {"category": category},
        {"$push": {"notices": notice}}
    )
    return redirect('/handle_notices')

@app.route('/remove_notices', methods=['POST'])
def remove_selected_notices():
    NoticeBoard = db["Notices"]
    selected_notices = request.form.getlist('remove')
    for category in NoticeBoard.find():
        for notice in category['notices']:
            if notice in selected_notices:
                NoticeBoard.update_one(
                    {"category": category['category']},
                    {"$pull": {"notices": notice}}
                )
    return redirect('/handle_notices')

@app.route('/handle_notices', methods=['GET'])
def handle_notices():
    NoticeBoard = db["Notices"]
    notices = list(NoticeBoard.find())

    return render_template('ChangeNoticeBoard.html', notices=notices)



@app.route('/ingredients', methods=['GET', 'POST'])
def ingredients():
    if request.method == 'POST':
        ingredients_str = request.form.get('ingredients')
        ingredients = ingredients_str.split(',')

        # Perform the poll logic based on ingredients and display the results
        results = db['Recipes'].find()
        matching_dishes = [document['dish_name'] for document in results if set(ingredients) <= set(document['ingredients'])]
 
        return render_template('ingredients.html', matching_dishes=matching_dishes)
    
    return render_template('ingredients.html')

def show_roadmaps():
    Roadmaps = DB["Roadmaps"]
    roadmaps = iter(Roadmaps.find())
    return roadmaps

def show_courses():
    Courses = DB["Courses"]
    courses = iter(Courses.find())
    return courses

def show_FreeStuff():
    FreeStuff = DB["FreeStuff"]
    items = iter(FreeStuff.find())
    return items

def create_roadmap(Title,Description,Image,Link):
    Roadmaps = DB["Roadmaps"]
    new_roadmap = {
        "Title": Title,
        "Description": Description,
        "Image" : Image,
        "Link": Link        
        }
    Roadmaps.insert_one(new_roadmap)

def create_FreeStuff(Title,Description,Image,Link):
    FreeStuff = DB["FreeStuff"]
    new_stuff = {
        "Title": Title,
        "Description": Description,
        "image" : Image,
        "Link": Link        
    }
    FreeStuff.insert_one(new_stuff)

def create_courses(Title,Description,Image,Link):
    Courses = DB["Courses"]
    new_course = {
        "Title": Title,
        "Description": Description,
        "image" : Image,
        "Link": Link        
    }
    Courses.insert_one(new_course)


@app.route('/Courses', methods=['GET', 'POST'])
def Courses():
    courses = show_courses()
    return render_template('Courses.html', courses=courses)

@app.route('/add_course', methods=['POST'])
def add_course():
    Title = request.form.get('course_name')
    Description = request.form.get('course_desc')
    Image = request.form.get('image_url')
    Link = request.form.get('course_url')
    create_courses(Title,Description,Image,Link)
    return redirect('/Courses')

@app.route('/Resources', methods=['GET', 'POST'])
def Resources():
    resources = show_FreeStuff()
    return render_template('Resources.html', resources=resources)

@app.route('/add_resource', methods=['POST'])
def add_resource():
    Title = request.form.get('resource_name')
    Description = request.form.get('resource_desc')
    Image = request.form.get('image_url')
    Link = request.form.get('resource_url')
    create_FreeStuff(Title,Description,Image,Link)
    return redirect('/Resources')

@app.route('/Roadmaps', methods=['GET', 'POST'])
def Roadmaps():
    roadmaps = show_roadmaps()
    return render_template('Roadmaps.html', roadmaps=roadmaps)

@app.route('/add_roadmap', methods=['POST'])
def add_roadmap():
    Title = request.form.get('roadmap_name')
    Description = request.form.get('roadmap_desc')
    Image = request.form.get('image_url')
    Link = request.form.get('roadmap_url')
    create_roadmap(Title,Description,Image,Link)
    return redirect('/Roadmaps')


if __name__ == '__main__':
    app.run()
