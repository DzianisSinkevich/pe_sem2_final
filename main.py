from fastapi import FastAPI
import uvicorn
from fastapi.responses import HTMLResponse

from data_creation import dc_main
# from data_preprocessing import dp_main
# from model_preparation import mp_main
# from model_testing import mt_main

app = FastAPI()


# Сообщение-заглушка для метода GET с инструкцией для метода POST
@app.get("/")
def get_root():
    dc_main()
    # dp_main()
    # mp_main()
    # mt_main()
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
    return HTMLResponse(content=responce_content, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
    