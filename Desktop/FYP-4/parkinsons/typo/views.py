from django.shortcuts import render
from django.http import HttpResponse
import csv
import io
# Create your views here.


def index(request):
    return render(request, 'index.html')


def dataset(request):
    return render(request, 'datasets.html')


def upload_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']

        # Decode the CSV file
        data_set = csv_file.read().decode('UTF-8')
        print(data_set)
        io_string = io.StringIO(data_set)

        # Parse the CSV file
        for row in csv.reader(io_string, delimiter=',', quotechar="|"):
            # Do something with each row of the CSV file
            pass

        return HttpResponse('File uploaded successfully.')
    else:
        return render(request, 'upload_csv.html')
