#Выгружаем библиотеки для работы
from decimal import Decimal
from fractions import Fraction

#Выгружаем код и других файлов
from .calc_config.configuration import multiple_replace
from .calc_config.formulas import puasson, bernulli, loc_laplas, int_laplas

#Отрисовка "Дано"
def con(n, p, q, pq):
    if pq == None:
        text = multiple_replace(r'''
    n = n_con%
    p = $\frac{p_con[0]}{p_con[1]}$%
    q = $\frac{q_con[0]}{q_con[1]}$
        ''', {'n_con': str(n), 'p_con[0]': str(p[0]), 'p_con[1]': str(p[1]), 'q_con[0]': str(q[0]), 'q_con[1]': str(q[1])})
    else:
        text = multiple_replace(r'''
    n = n_con%
    p = pq[0] = $\frac{p_con[0]}{p_con[1]}$%
    q = pq[1] = $\frac{q_con[0]}{q_con[1]}$
        ''', {'n_con': str(n), 'p_con[0]': str(p[0]), 'p_con[1]': str(p[1]), 'q_con[0]': str(q[0]), 'q_con[1]': str(q[1]), 'pq[0]': str(pq[0]), 'pq[1]': str(pq[1])})

    return text.split('%')

    # plt.figure(figsize=(3, 2))
    # plt.text(-0.15, 1.12, 'Дано', fontsize=20, fontweight='bold', ha='left', va='top')
    # plt.text(-0.26, 1.05, text, fontsize=20, ha='left', va='top')
    # plt.axis('off')

    # plt.savefig('main_page/static/main_page/img/condition.png', format='png')

#Отрисовка "Найти"
def find(n, m1, m2, sign1, sign2):
    if sign1 not in (None, '='):
        sign1 = sign1 if sign1 in ('<', '>') else r'\leq' if sign1 == '<=' else r'\geq'

    if sign2 not in (None, '='):
        sign2 = sign2 if sign2 in ('<', '>') else r'\leq' if sign2 == '<=' else r'\geq'

    if sign2 != None:
        text = multiple_replace(r'  $P_{n}(m1 sign1 m sign2 m2)$', {'sign1': sign1, 'n': str(n), 'm1': str(m1), 'sign2': sign2, 'm2': str(m2)})
    elif sign1 == '=':
        text = multiple_replace(r'  $P_{n}(m)$', {'n': str(n), 'm': str(m1)})
    else:
        text = multiple_replace(r'  $P_{n}(m sign m_value)$', {'sign': sign1, 'n': str(n), 'm_value': str(m1)})

    return text

    # plt.figure(figsize=(3, 0.5))
    # plt.text(-0.15, 1.15, text + '-?', fontsize=15, ha='left', va='top')
    # plt.axis('off')

    # plt.savefig('main_page/static/main_page/img/find.png', format='png')

#Отрисовка "Решения"
def sol(n, m1, m2, sign1, sign2, p):
    # Форматируем вероятность
    pq = None
    if '/' in p:
        p = tuple(map(int, p.split('/')))
        q = (p[1] - p[0], p[1])
    else:
        pq = (float(p), float(1 - Decimal(str(p))))
        p = tuple(map(int, str(Fraction(pq[0]).limit_denominator()).split('/')))
        q = tuple(map(int, str(Fraction(pq[1]).limit_denominator()).split('/')))

    con_text = con(n, p, q, pq)
    find_text = find(n, m1, m2, sign1, sign2)

    npq = Decimal(n) * Decimal(p[0]) / Decimal(p[1]) * Decimal(q[0]) / Decimal(q[1])
    np = Decimal(n) * Decimal(p[0]) / Decimal(p[1])

    if np < 10 and p[0] / p[1] < 0.01:
        f = puasson(n, m1, m2, p, np, sign1, sign2, find_text, pq)
    elif npq >= 10:
        if sign1 == '':
            f = loc_laplas(n, m1, p, q, npq, np, find_text, pq)
        else:
            f = int_laplas(n, m1, m2, p, q, npq, np, find_text, sign1, pq)
    elif npq < 10 and (p[0] / p[1] > 0.01 or n < 99):
        f = bernulli(n, m1, m2, sign1, sign2, p, q, npq, pq, find_text)
    else:
        f = '\n    Ошибка'

    return {'con': con_text, 'find': find_text, 'sol': f[:-1], 'ans': f[-1]}

    # plt.figure(figsize=(15, 20))
    # plt.text(-0.13, 1.15, '    Решение', fontsize=20, fontweight='bold', ha='left', va='top')
    # plt.text(-0.15, 1.14, f, fontsize=20, ha='left', va='top')
    # plt.axis('off')

    # plt.savefig('main_page/static/main_page/img/solution.png', format='png')