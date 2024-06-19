from pathlib import Path
from flask import Flask,render_template
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_cors import CORS
from apps.config import config

db = SQLAlchemy()
csrf=CSRFProtect()

def  create_app(config_key):
    app=Flask(__name__)

    app.config.from_object(config[config_key])
    
    csrf.init_app(app)
    db.init_app(app)
    Migrate(app, db)
    CORS(app)

    CORS(app, resources={
        r"/unocar/*": {"origins": ["http://localhost/index.do?mCode=M060000", "http://another-domain.com"]},
        r"/unofarm/*": {"origins": "*"}
    })

    from apps.unoFarm import views as uno_views
    app.register_blueprint(uno_views.unofarm, url_prefix="/unofarm") 

    from apps.unoCar import views as car_views
    app.register_blueprint(car_views.unocar, url_prefix="/unocar") 

    from apps.controlapi import views as control_views
    app.register_blueprint(control_views.controlapi, url_prefix="/control")
    

    @app.route("/")
    def index():
         return render_template("index.html")
    
    
    return app
    
    # app.register_error_handler(404, page_not_found)
    # app.register_error_handler(Exception, internal_server_error)



# def page_not_found(e):
#     """404 Not Found"""
#     return render_template("404.html"), 404 

# def internal_server_error(e):
#     """500 Internal Server Error"""
#     return render_template("500.html"), 500
    
