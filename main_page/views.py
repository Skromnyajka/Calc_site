from django.shortcuts import render

from .calc_file import sol

def index(request):
    return render(request, 'main.html')

def int_index(request):
    return render(request, 'interval_page.html')

def done(request):
    if request.GET['but'] == 'zn':
        n = int(request.GET['n'])
        m1 = int(request.GET['m1'])
        m2 = None
        sign1 = request.GET['sign1']
        sign2 = None
        p = request.GET['p']
    else:
        n = int(request.GET['n'])
        m1 = int(request.GET['m1'])
        m2 = int(request.GET['m2'])
        sign1 = '<='
        sign2 = '<='
        p = request.GET['p']
    
    data = sol(n, m1, m2, sign1, sign2, p)
    return render(request, 'done.html', context=data)

def code(request):
    return render(request, 'code.html')
