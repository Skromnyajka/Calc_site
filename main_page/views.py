from django.shortcuts import render

from .calc_file import sol

def index(request):
    return render(request, 'main.html')

def done(request):
    n = int(request.GET['n'])
    m1 = int(request.GET['m1'])
    m2 = None
    sign1 = request.GET['sign1']
    sign2 = None
    p = request.GET['p']
    
    data = sol(n, m1, m2, sign1, sign2, p)
    return render(request, 'done.html', context=data)


