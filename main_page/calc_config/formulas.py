#Выгружаем библиотеки для работы
from math import prod, gcd, exp, factorial, sqrt
from decimal import Decimal

#Выгружаем код и других файлов
from .configuration import multiple_replace, sum_fract
from .tables_data import loc_data, int_data

#Решение по ф-ле Бернулли
def bernulli(n, m1, m2, sign1, sign2, p, q, npq, pq, find_text):
    print('bernulli')

    f1 = f'''
    npq = {n}''' + multiple_replace(r' $\cdot \ \frac{p[0]}{p[1]} \cdot \frac{q[0]}{q[1]}$ ', {'p[0]': str(p[0]), 'p[1]': str(p[1]), 'q[0]': str(q[0]), 'q[1]': str(q[1])} if pq == None else {r'\frac{p[0]}{p[1]}': str(pq[0]), r'\frac{q[0]}{q[1]}': str(pq[1])}) \
         + f'= {npq} < 10 - применяем ф-лу Бернулли'

    f2 = ''
    f21 = ''
    f22 = ''
    f23 = ''
    f3 = ''
    ln = []
    ld = []

    if sign1 == '=':
        f2 = r'$P_n(m) = C^m_n p^m q^{n-m}$'

        f21 += multiple_replace(r'find = $C^{m}_{n} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$', {'n': str(n), 'm': str(m1), 'p[0]': str(p[0]), 'p[1]': str(p[1]), 'q[0]': str(q[0]), 'q[1]': str(q[1]), 'find': find_text}) \
               + multiple_replace \
            (r' = $\frac{n!}{m!n-m!} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ =  ', {'q[0]': str(q[0]), 'p[0]': str(p[0]), 'q[1]': str(q[1]), 'p[1]': str(p[1]), 'n-m': str( n -m1), 'n': str(n), 'm': str(m1)})

        res_p = (p[0]**m1, p[1]**m1)
        res_q = (q[0]**(n-m1), q[1]**(n-m1))

        fact1 = [i for i in range(2, n+ 1)]
        if m1 < n - m1:
            fact1 = [i for i in fact1 if i not in [i for i in range(2, n - m1 + 1)]]
            fact2 = [i for i in range(2, m1 + 1)] if n - m1 + 1 > 2 else [1]
        else:
            fact1 = [i for i in fact1 if i not in [i for i in range(2, m1 + 1)]]
            fact2 = [i for i in range(2, n - m1 + 1)] if n - m1 + 1 > 2 else [1]
        fact1, fact2 = [1] if len(fact1) == 0 else fact1, [1] if len(fact2) == 0 else fact2

        f22 += multiple_replace(
            r'$\frac{fact1}{fact2} \cdot \frac{res_p[0]}{res_p[1]} \cdot \frac{res_q[0]}{res_q[1]}$ = ',
            {'fact1': r'\cdot'.join(map(str, fact1)), 'fact2': r'\cdot'.join(map(str, fact2)),
             'res_p[0]': str(res_p[0]), 'res_p[1]': str(res_p[1]), 'res_q[0]': str(res_q[0]),
             'res_q[1]': str(res_q[1])})

        numerator = prod(fact1 + [res_q[0]] + [res_p[0]])
        denominator = prod(fact2 + [res_q[1]] + [res_p[1]])
        f_res = round(Decimal(numerator)/Decimal(denominator), 3)
        f_res = f_res if f_res != int(f_res) else 0
        nod = gcd(numerator, denominator)
        f23 += multiple_replace(r'$\frac{numerator}{denominator}$ = ',
                                {'numerator': str(numerator), 'denominator': str(denominator)}) \
               + multiple_replace(r'$\frac{numerator}{denominator} \approx$', {'numerator': str(Decimal(numerator // nod)),
                                                                       'denominator': str(
                                                                           Decimal(denominator // nod))}) + f' {f_res}'

    elif sign1 == '<=' or sign1 == '<':
        f2 = r'$P_n(m) = C^m_n p^m q^{n-m}$'

        f2 += f'{find_text} = '
        for i in range(m1 + (sign1 == '<=')):
            if i == (m1 - 1 + (sign1 == '<=')):
                f21 += multiple_replace(r'$C^{m}_{n} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ = ',
                                        {'n': str(n), 'm': str(i), 'p[0]': str(p[0]), 'p[1]': str(p[1]),
                                         'q[0]': str(q[0]), 'q[1]': str(q[1])})
            elif i == 0:
                f21 += multiple_replace(r'$C^{m}_{n} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n}$ + ',
                                        {'n': str(n), 'm': str(i), 'p[0]': str(p[0]), 'p[1]': str(p[1]),
                                         'q[0]': str(q[0]), 'q[1]': str(q[1])})
            else:
                f21 += multiple_replace(r'$C^{m}_{n} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ + ',
                                        {'n': str(n), 'm': str(i), 'p[0]': str(p[0]), 'p[1]': str(p[1]),
                                         'q[0]': str(q[0]), 'q[1]': str(q[1])})

            if i == (m1 - 1 + (sign1 == '<=')):
                f22 += multiple_replace(
                    r'$\frac{n!}{m!n-m!} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ = ',
                    {'q[0]': str(q[0]), 'p[0]': str(p[0]), 'q[1]': str(q[1]), 'p[1]': str(p[1]), 'n-m': str(n - i),
                     'n': str(n), 'm': str(i)})
            else:
                f22 += multiple_replace(
                    r'$\frac{n!}{m!n-m!} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ + ',
                    {'q[0]': str(q[0]), 'p[0]': str(p[0]), 'q[1]': str(q[1]), 'p[1]': str(p[1]), 'n-m': str(n - i),
                     'n': str(n), 'm': str(i)})

            res_p = (p[0] ** i, p[1] ** i)
            res_q = (q[0] ** (n - i), q[1] ** (n - i))

            fact1 = [j for j in range(2, n + 1)]

            if i < n - i:
                fact1 = [j for j in fact1 if j not in [j for j in range(2, n - i + 1)]]
                fact2 = [j for j in range(2, i + 1)] if n - i + 1 > 2 else [1]
            else:
                fact1 = [j for j in fact1 if j not in [j for j in range(2, i + 1)]]
                fact2 = [j for j in range(2, n - i + 1)] if n - i + 1 > 2 else [1]
            fact1, fact2 = [1] if len(fact1) == 0 else fact1, [1] if len(fact2) == 0 else fact2

            if i == (m1 - 1 + (sign1 == '<=')):
                f23 += multiple_replace(
                    r'$\frac{fact1}{fact2} \cdot \frac{res_p[0]}{res_p[1]} \cdot \frac{res_q[0]}{res_q[1]}$ = ',
                    {'fact1': r'\cdot'.join(map(str, fact1)), 'fact2': r'\cdot'.join(map(str, fact2)),
                     'res_p[0]': str(res_p[0]), 'res_p[1]': str(res_p[1]), 'res_q[0]': str(res_q[0]),
                     'res_q[1]': str(res_q[1])})
            else:
                f23 += multiple_replace(
                    r'$\frac{fact1}{fact2} \cdot \frac{res_p[0]}{res_p[1]} \cdot \frac{res_q[0]}{res_q[1]}$ + ',
                    {'fact1': r'\cdot'.join(map(str, fact1)), 'fact2': r'\cdot'.join(map(str, fact2)),
                     'res_p[0]': str(res_p[0]), 'res_p[1]': str(res_p[1]), 'res_q[0]': str(res_q[0]),
                     'res_q[1]': str(res_q[1])})

            numerator = prod(fact1 + [res_q[0]] + [res_p[0]])
            denominator = prod(fact2 + [res_q[1]] + [res_p[1]])
            f_res = round(Decimal(numerator)/Decimal(denominator), 3)
            f_res = f_res if f_res != int(f_res) else 0
            if i == (m1 - 1 + (sign1 == '<=')):
                f3 += multiple_replace(r'$\frac{num}{den}$ = ', {'num': str(numerator), 'den': str(denominator)})
            else:
                f3 += multiple_replace(r'$\frac{num}{den}$ + ', {'num': str(numerator), 'den': str(denominator)})
            ln.append(numerator)
            ld.append(denominator)

        ln, ld = sum_fract(ln, ld)
        nod = gcd(ln, ld)
        f3 += multiple_replace(r'$\frac{num}{den}$ = ', {'num': str(ln), 'den': str(ld)})
        f3 += multiple_replace(r'$\frac{num}{den}$',
                               {'num': str(Decimal(ln // nod)), 'den': str(Decimal(ld // nod))}) + f' {f_res}'

    elif sign1 == '>=' or sign1 == '>':
        f2 = r'$P_n(m) = C^m_n p^m q^{n-m}$'

        f2 += f'{find_text} = '
        for i in range(m1 + (sign1 == '>='), n + 1):
            if i == (n - 1 + (sign1 == '>=')):
                f21 += multiple_replace(r'$C^{m}_{n} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ = ',
                                        {'n': str(n), 'm': str(i), 'p[0]': str(p[0]), 'p[1]': str(p[1]),
                                         'q[0]': str(q[0]), 'q[1]': str(q[1])})
            elif i == 0:
                f21 += multiple_replace(r'$C^{m}_{n} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n}$ + ',
                                        {'n': str(n), 'm': str(i), 'p[0]': str(p[0]), 'p[1]': str(p[1]),
                                         'q[0]': str(q[0]), 'q[1]': str(q[1])})
            else:
                f21 += multiple_replace(r'$C^{m}_{n} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ + ',
                                        {'n': str(n), 'm': str(i), 'p[0]': str(p[0]), 'p[1]': str(p[1]),
                                         'q[0]': str(q[0]), 'q[1]': str(q[1])})

            if i == (n - 1 + (sign1 == '>=')):
                f22 += multiple_replace(
                    r'$\frac{n!}{m!n-m!} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ = ',
                    {'q[0]': str(q[0]), 'p[0]': str(p[0]), 'q[1]': str(q[1]), 'p[1]': str(p[1]), 'n-m': str(n - i),
                     'n': str(n), 'm': str(i)})
            else:
                f22 += multiple_replace(
                    r'$\frac{n!}{m!n-m!} \cdot (\frac{p[0]}{p[1]})^{m} \cdot (\frac{q[0]}{q[1]})^{n-m}$ + ',
                    {'q[0]': str(q[0]), 'p[0]': str(p[0]), 'q[1]': str(q[1]), 'p[1]': str(p[1]), 'n-m': str(n - i),
                     'n': str(n), 'm': str(i)})

            res_p = (p[0] ** i, p[1] ** i)
            res_q = (q[0] ** (n - i), q[1] ** (n - i))

            fact1 = [j for j in range(2, n + 1)]

            if i < n - i:
                fact1 = [j for j in fact1 if j not in [j for j in range(2, n - i + 1)]]
                fact2 = [j for j in range(2, i + 1)] if n - i + 1 > 2 else [1]
            else:
                fact1 = [j for j in fact1 if j not in [j for j in range(2, i + 1)]]
                fact2 = [j for j in range(2, n - i + 1)] if n - i + 1 > 2 else [1]
            fact1, fact2 = [1] if len(fact1) == 0 else fact1, [1] if len(fact2) == 0 else fact2

            if i == (n - 1 + (sign1 == '>=')):
                f23 += multiple_replace(
                    r'$\frac{fact1}{fact2} \cdot \frac{res_p[0]}{res_p[1]} \cdot \frac{res_q[0]}{res_q[1]}$ = ',
                    {'fact1': r'\cdot'.join(map(str, fact1)), 'fact2': r'\cdot'.join(map(str, fact2)),
                     'res_p[0]': str(res_p[0]), 'res_p[1]': str(res_p[1]), 'res_q[0]': str(res_q[0]),
                     'res_q[1]': str(res_q[1])})
            else:
                f23 += multiple_replace(
                    r'$\frac{fact1}{fact2} \cdot \frac{res_p[0]}{res_p[1]} \cdot \frac{res_q[0]}{res_q[1]}$ + ',
                    {'fact1': r'\cdot'.join(map(str, fact1)), 'fact2': r'\cdot'.join(map(str, fact2)),
                     'res_p[0]': str(res_p[0]), 'res_p[1]': str(res_p[1]), 'res_q[0]': str(res_q[0]),
                     'res_q[1]': str(res_q[1])})

            numerator = prod(fact1 + [res_q[0]] + [res_p[0]])
            denominator = prod(fact2 + [res_q[1]] + [res_p[1]])
            f_res = round(Decimal(numerator)/Decimal(denominator), 3)
            f_res = f_res if f_res != int(f_res) else 0
            if i == (n - 1 + (sign1 == '>=')):
                f3 += multiple_replace(r'$\frac{num}{den}$ = ', {'num': str(numerator), 'den': str(denominator)})
            else:
                f3 += multiple_replace(r'$\frac{num}{den}$ + ', {'num': str(numerator), 'den': str(denominator)})
            ln.append(numerator)
            ld.append(denominator)

        ln, ld = sum_fract(ln, ld)
        nod = gcd(ln, ld)
        f3 += multiple_replace(r'$\frac{num}{den}$ = ', {'num': str(ln), 'den': str(ld)})
        f3 += multiple_replace(r'$\frac{num}{den} \approx$',
                               {'num': str(Decimal(ln // nod)), 'den': str(Decimal(ld // nod))}) + f' {f_res}'

    formula = [f1, '', f2, '', f21 + f22 + f23 + f3, f_res]
    return formula

#Решение по ф-ле Пуассона
def puasson(n, m1, m2, p, np, sign1, sign2, find_text, pq):
    print('puasson')

    f1 = f'''
    np = {n} ''' + multiple_replace(r'$\cdot \ \frac{p[0]}{p[1]}$',
                                    {'p[0]': str(p[0]), 'p[1]': str(p[1])} if pq == None else {
                                        r'\frac{p[0]}{p[1]}': str(pq[0]), r'\frac{q[0]}{q[1]}': str(pq[1])}) \
         + f''' = {np} < 10 - применяем ф-лу Пуассона'''
    f_2 = r'$P_n(m) \approx \frac{\lambda^m}{m!} \cdot e^{-\lambda}\lambda$ = np = ' + str(np)

    if sign1 == '<=' and sign2 == '<=':
        sign1 = r'\leq' if sign1 == '<=' else '<'
        f2 = r'find $\approx$ '.replace('find', find_text)

        for i in range(m1, m2 + (sign1 == r'\leq')):
            if i == (m2 - 1 + (sign1 == r'\leq')):
                f2 += multiple_replace(r'$P_{n}({m}) \approx$ ', {'n': str(n), 'm': str(i)})
            else:
                f2 += multiple_replace(r'$P_{n}({m})$ + ', {'n': str(n), 'm': str(i)})
        f2 += r'$\approx$ '
        for i in range(m1, m2 + (sign1 == r'\leq')):
            if i == (m2 - 1 + (sign1 == r'\leq')):
                f2 += multiple_replace(r'$\frac{np^{m}}{m!} \cdot e^{-np} \approx$', {'np': str(np), 'm': str(i)})
            else:
                f2 += multiple_replace(r'$\frac{np^{m}}{m!} \cdot e^{-np}$ + ', {'np': str(np), 'm': str(i)})
        f2 += r'$\approx$ ('
        res_frac = Decimal('0')
        for i in range(m1, m2 + (sign1 == r'\leq')):
            frac = round(round(np ** i, 4) / factorial(i), 4)
            res_frac += Decimal(str(frac))
            if i == (m2 - 1 + (sign1 == r'\leq')):
                f2 += r'frac)'.replace('frac', str(int(frac) if int(frac) == frac else frac))
            else:
                f2 += r'frac + '.replace('frac', str(int(frac) if int(frac) == frac else frac))
        f2 += r' $\cdot \ e^{-np} \approx$ '.replace('np', str(np))
        result = float(res_frac / Decimal(exp(1)) ** Decimal(np))
        result = int(result) if int(result) == result else round(result, 5)
        f2 += multiple_replace(r'$\approx res_frac \cdot e^{-np} \approx \frac{res_frac}{e^{np}} \approx$ result',
                               {'result': str(result), 'np': str(np),
                                'res_frac': str(int(res_frac) if int(res_frac) == res_frac else res_frac)})
        f3 = str(int(result) if int(result) == result else result)
    elif sign1 == '>' or sign1 == '>=':
        f2 = multiple_replace(r'$P_{n}({m sign m_value}) \approx$ ',
                              {'sign': sign1 if sign1 == '>' else r'\geq', 'n': str(n), 'm_value': str(m1)})
        sign1 = r'\leq' if sign1 == '>' else '<'
        f2 += multiple_replace(r'''1 - $P_{n}({m sign m_value}) \approx$ 1 - (''',
                               {'sign': sign1, 'n': str(n), 'm_value': str(m1)})

        for i in range(m1 + (sign1 == r'\leq')):
            if i == (m1 - 1 + (sign1 == r'\leq')):
                f2 += multiple_replace(r'$P_{n}({m})) \approx$ ', {'n': str(n), 'm': str(i)})
            else:
                f2 += multiple_replace(r'$P_{n}({m})$ + ', {'n': str(n), 'm': str(i)})
        f2 += r'$\approx 1 - ($'
        for i in range(m1 + (sign1 == r'\leq')):
            if i == (m1 - 1 + (sign1 == r'\leq')):
                f2 += multiple_replace(r'$\frac{np^m}{m!} \cdot e^{-np}) \approx$ ', {'np': str(np), 'm': str(i)})
            else:
                f2 += multiple_replace(r'$\frac{np^m}{m!} \cdot e^{-np}$ + ', {'np': str(np), 'm': str(i)})
        f2 += r'$\approx 1 - ($'
        res_frac = Decimal('0')
        for i in range(m1 + (sign1 == r'\leq')):
            frac = round(round(np ** i, 4) / factorial(i), 5)
            res_frac += Decimal(str(frac))
            if i == (m1 - 1 + (sign1 == r'\leq')):
                f2 += r'frac)'.replace('frac', str(int(frac) if int(frac) == frac else frac))
            else:
                f2 += r'frac + '.replace('frac', str(int(frac) if int(frac) == frac else frac))
        f2 += r' $\cdot \ e^{-np} \approx$'.replace('np', str(np))
        result = round(float(1 - res_frac / Decimal(exp(1)) ** Decimal(np)), 5)
        f2 += multiple_replace(
            r'$\approx 1 - res_frac \cdot e^{-np} \approx 1 - \frac{res_frac}{e^{np}} \approx$ result',
            {'result': str(result), 'np': str(np), 'res_frac': str(res_frac)})
        f3 = str(int(result) if int(result) == result else result)

    elif sign1 == '<' or sign1 == '<=':
        sign1 = r'\leq' if sign1 == '<=' else '<'
        f2 = multiple_replace(r'$P_{n}({m sign m_value}) \approx$ ', {'sign': sign1, 'n': str(n), 'm_value': str(m1)})

        for i in range(m1 + (sign1 == r'\leq')):
            if i == (m1 - 1 + (sign1 == r'\leq')):
                f2 += multiple_replace(r'$P_{n}({m}) \approx$ ', {'n': str(n), 'm': str(i)})
            else:
                f2 += multiple_replace(r'$P_{n}({m})$ + ', {'n': str(n), 'm': str(i)})
        f2 += r'$\approx$ '
        for i in range(m1 + (sign1 == r'\leq')):
            if i == (m1 - 1 + (sign1 == r'\leq')):
                f2 += multiple_replace(r'$\frac{np^m}{m!} \cdot e^{-np} \approx$', {'np': str(np), 'm': str(i)})
            else:
                f2 += multiple_replace(r'$\frac{np^m}{m!} \cdot e^{-np}$ + ', {'np': str(np), 'm': str(i)})
        f2 += r'$\approx$ ('
        res_frac = Decimal('0')
        for i in range(m1 + (sign1 == r'\leq')):
            frac = round(round(np ** i, 4) / factorial(i), 4)
            res_frac += Decimal(str(frac))
            if i == (m1 - 1 + (sign1 == r'\leq')):
                f2 += r'frac)'.replace('frac', str(int(frac) if int(frac) == frac else frac))
            else:
                f2 += r'frac + '.replace('frac', str(int(frac) if int(frac) == frac else frac))
        f2 += r' $\cdot \ e^{-np} \approx$ '.replace('np', str(np))
        result = round(float(res_frac / Decimal(exp(1)) ** Decimal(np)), 5)
        f2 += multiple_replace(r'$\approx res_frac \cdot e^{-np} \approx \frac{res_frac}{e^{np}} \approx$ result',
                               {'result': str(result), 'np': str(np), 'res_frac': str(res_frac)})
        f3 = str(int(result) if int(result) == result else result)

    else:
        frac = Decimal(round(round(np ** m1, 4) / factorial(m1), 4))
        result = round(float(frac / Decimal(exp(1)) ** Decimal(np)), 5)

        f2 = multiple_replace(
            r'''find $\approx \frac{np^m}{m!} \cdot e^{-np} \approx frac_form \cdot e^{-np} \approx \frac{frac_form}{e^{np}} \approx result$''',
            {'np': str(np), 'find': find_text, 'frac_form': str(int(frac) if int(frac) == frac else frac),
             'result': str(result)})
        f3 = str(int(result) if int(result) == result else result)

    formula = [f1, '', f_2, '', f2, f3]

    return formula

#Решение по локальной ф-ле Лапласа
def loc_laplas(n, m1, p, q, npq, np, find_text, pq):
    print('loc_laplas')

    f1 = f'''
    npq = {n} ''' + multiple_replace(r'$\cdot \ \frac{p[0]}{p[1]} \cdot \frac{q[0]}{q[1]}$',
                                     {'p[0]': str(p[0]), 'p[1]': str(p[1]), 'q[0]': str(q[0]),
                                      'q[1]': str(q[1])} if pq == None else {r'\frac{p[0]}{p[1]}': str(pq[0]),
                                                                             r'\frac{q[0]}{q[1]}': str(pq[1])}) \
         + f' = {npq}' + r'$\geq$' + '10 - применяем ф-лу Лапласа'

    f_2 = r'1) $x = \frac{m-np}{\sqrt{npq}}$'
    f2 = multiple_replace(r'x = $\frac{m-n \cdot p}{\sqrt{npq}}$ = ', {'m': str(m1), 'npq': str(npq), 'n': str(n),
                                                                        'p': str(
                                                                            Decimal(p[0]) / Decimal(p[1]) if Decimal(
                                                                                p[0]) / Decimal(p[1]) == round(
                                                                                Decimal(p[0]) / Decimal(p[1]),
                                                                                5) else multiple_replace(
                                                                                r'\frac{p[0]}{p[1]}',
                                                                                {'p[0]': str(p[0]),
                                                                                 'p[1]': str(p[1])}))})
    frac1 = Decimal(m1) - Decimal(np)
    frac2 = round(sqrt(npq), 5)
    x = round(Decimal(frac1) / Decimal(frac2), 2)
    f2 += multiple_replace(r'$\frac{frac1}{frac2}$ = ', {'frac1': str(int(frac1) if int(frac1) == frac1 else frac1),
                                                         'frac2': str(int(frac2) if int(
                                                             frac2) == frac2 else frac2)}) + str(x)

    f2 += r'2) $\varphi(x) = \frac{1}{\sqrt{2\pi}} \cdot e^{\frac{-x^2}{2}}$ - табличное значение' + multiple_replace(
        r'(функция $\varphi(x)$ - четная, поэтому x_val1 = x_val2)',
        {'x_val1': str(x), 'x_val2': str(x * -1)}) if x < 0 else ''
    f2 += r'$\varphi(x)$ = ' + r'$\varphi(x)$ = '.replace('x', str(x if x >= 0 else x * -1))
    fux = float(x if x >= 0 else x * -1)
    fux = loc_data(round(fux, 2), int(fux * 100 % 10 // 1)) if fux <= 5 else 0
    f2 += str(fux) + r' - при $x\geq5, \ \varphi(x) \rightarrow$0'

    f2 += r'3) $P_n(m) \approx \frac{\varphi(x)}{\sqrt{npq}}$'
    result = fux / frac2
    result = result if int(result) == result else round(result, 5)
    f2 += multiple_replace(r'find $\approx \frac{fux}{sqrt} \approx result$', {'find': find_text, 'fux': str(fux),
                                                                               'sqrt': str(int(frac2) if int(
                                                                                   frac2) == frac2 else frac2),
                                                                               'result': str(int(result) if int(
                                                                                   result) == result else result)})

    f3 = str(int(result) if int(result) == result else result)

    formula = [f1, '', f_2, '', f2, f3]
    return formula

#Решение по интегральной ф-ле Лапласа
def int_laplas(n, m1, m2, p, q, npq, np, find_text, sign1, pq):
    print('int_laplas')

    if m2 == None and (sign1 == '<=' or sign1 == '<'):
        m2 = m1
        m1 = 0
    elif m2 == None and (sign1 == '>=' or sign1 == '>'):
        m2 = n

    f1 = f'''
    npq = {n} ''' + multiple_replace(r'$\cdot \ \frac{p[0]}{p[1]} \cdot \frac{q[0]}{q[1]}$',
                                     {'p[0]': str(p[0]), 'p[1]': str(p[1]), 'q[0]': str(q[0]),
                                      'q[1]': str(q[1])} if pq == None else {r'\frac{p[0]}{p[1]}': str(pq[0]),
                                                                             r'\frac{q[0]}{q[1]}': str(pq[1])}) \
         + f' = {npq}' + r'$\geq$' + '10 - применяем интегральную ф-лу Лапласа'

    f_2 = r'1) $x_1 = \frac{m_1-np}{\sqrt{npq}}$'
    f2 = multiple_replace(r'$x_1 = \frac{m_1-n \cdot p}{\sqrt{npq}}$ = ',
                           {'m_1': str(m1), 'npq': str(npq), 'n': str(n), 'p': str(
                               Decimal(p[0]) / Decimal(p[1]) if Decimal(p[0]) / Decimal(p[1]) == round(
                                   Decimal(p[0]) / Decimal(p[1]), 5) else multiple_replace(r'\frac{p[0]}{p[1]}',
                                                                                           {'p[0]': str(p[0]),
                                                                                            'p[1]': str(p[1])}))})
    frac1 = Decimal(m1 - np)
    frac2 = round(sqrt(npq), 5)
    x1 = round(Decimal(frac1) / Decimal(frac2), 2)
    f2 += multiple_replace(r'$\frac{frac1}{frac2}$ = ', {'frac1': str(int(frac1) if int(frac1) == frac1 else frac1),
                                                         'frac2': str(int(frac2) if int(
                                                             frac2) == frac2 else frac2)}) + str(x1)

    f2 += r'2) $x_2 = \frac{m_2-np}{\sqrt{npq}}$'
    f2 += multiple_replace(r'$x_2 = \frac{m_2-n \cdot p}{\sqrt{npq}}$ = ',
                           {'m_2': str(m2), 'npq': str(npq), 'n': str(n), 'p': str(
                               Decimal(p[0]) / Decimal(p[1]) if Decimal(p[0]) / Decimal(p[1]) == round(
                                   Decimal(p[0]) / Decimal(p[1]), 5) else multiple_replace(r'\frac{p[0]}{p[1]}',
                                                                                           {'p[0]': str(p[0]),
                                                                                            'p[1]': str(p[1])}))})
    frac1 = Decimal(m2 - np)
    frac2 = round(sqrt(npq), 5)
    x2 = round(Decimal(frac1) / Decimal(frac2), 2)
    f2 += multiple_replace(r'$\frac{frac1}{frac2}$ = ', {'frac1': str(int(frac1) if int(frac1) == frac1 else frac1),
                                                         'frac2': str(int(frac2) if int(
                                                             frac2) == frac2 else frac2)}) + str(x2)

    f2 += r'3) $\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_0^xe^{\frac{-t^2}{2}}$ - табличное значение'
    x1_flag = x1 < 0
    f2 += r'$\Phi(x_1)$ = '
    f2 += r'$-\Phi(x_1)$ = '.replace('x_1', str(x1 * -1)) if x1_flag else r'$\Phi(x_1)$ = '.replace('x_1', str(x1))
    fux1 = float(x1 * -1 if x1_flag else x1)
    fux1 = int_data(round(fux1, 2), int(fux1 * 100 % 10 // 1)) if fux1 < 5 else 0.5
    f2 += str(fux1) + r' - при $x\geq5, \ \varphi(x) \rightarrow$0.5'
    x2_flag = x2 < 0
    f2 += r'$\Phi(x_2)$ = '
    f2 += r'$-\Phi(x_2)$ = '.replace('x_2', str(x2 * -1)) if x2_flag else r'$\Phi(x_2)$ = '.replace('x_2', str(x2))
    fux2 = float(x2 * -1 if x2_flag else x2)
    fux2 = int_data(round(fux2, 2), int(fux2 * 100 % 10 // 1)) if fux2 < 5 else 0.5
    f2 += str(fux2) + r' - при $x\geq5, \ \varphi(x) \rightarrow$0.5'

    f2 += r'4) $P_n(m_1 \leqslant m \leqslant m_2) \ \approx \ \Phi(x_2)-\Phi(x_1)$'
    x1_res = ('+' if x1_flag else '-') + str(fux1)
    x2_res = ('-' if x2_flag else '') + str(fux2)
    result = Decimal(float(x2_res) + float(x1_res))
    result = result if int(result) == result else round(result, 5)
    f2 += multiple_replace(r'find $\approx x2x1 \approx result$',
                           {'find': find_text, 'x2': x2_res, 'x1': x1_res, 'result': str(result)})

    f3 = str(result)

    formula = [f1, '', f_2, '', f2, f3]
    return formula