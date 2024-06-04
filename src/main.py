from fastapi import FastAPI
import uvicorn
from fastapi.responses import HTMLResponse

from data_creation import dc_main
# from data_preprocessing import dp_main
# from model_preparation import mp_main
# from model_testing import mt_main

import logging
import configparser
app = FastAPI()

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/log.log')
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

settings = configparser.ConfigParser()
settings.read('config/settings.ini')


# Сообщение-заглушка для метода GET с инструкцией для метода POST
@app.get("/")
def get_root():
    logger.info("START")
    dc_main()
    # dp_main()
    # mp_main()
    # mt_main()
    logger.info("FINISH\n")
    logger.info("START results writing")
    with open('results/results', 'r') as f:
        data = f.read()
    print(data)
    data = data.replace('\n', '<br />')
    responce_content = """
    <!DOCTYPE html>”
    <header>Model results.</header>
    <body><p>
    """ + data + """
    </p></body>
    """
    logger.info("FINISH results writing")
    return HTMLResponse(content=responce_content, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
