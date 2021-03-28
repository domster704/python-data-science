from flask import Flask, request

from NeuroLand.programs_parser import domofond_parser

app = Flask(__name__)
url = None


@app.route("/")
def hello123():
	return 'xd'


# @app.route("/", methods=["GET"])
# def hello():
# 	data = domofond_parser.get_data_by_link(url)
# 	print(data)
# 	from new_tens import getDataFromReadyNeural
# 	per = getDataFromReadyNeural(data)
# 	print(per)
# 	return str(per)

@app.route('/', methods=["POST"])
def doit():
	global url
	request_data = request.form.to_dict()
	for k, v in request_data.items():
		url = k
	print(url)


if __name__ == "__main__":
	app.run(port=3000)
