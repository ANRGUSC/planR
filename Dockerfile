FROM python:3

ADD campus_digital_twin /
ADD campus_gym /
ADD data_generator /
ADD input_files /
ADD main.py /
ADD requirements.txt /
RUN pip install -r requirements.txt

CMD [ "python", "./main.py" ]