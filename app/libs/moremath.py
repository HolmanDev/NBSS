import numpy

# Ellipse integral with resolution n
def ellipseIntegral(k, n):
    value = 0
    i = 0
    halfpi = numpy.pi * 0.5
    d = halfpi / n
    while i < halfpi:
        trig = numpy.sin(i)
        value += numpy.sqrt(1-k*trig*trig) * d
        i += d
    return value

# arc = a * E(amp|k). k = m. second kind
def ellipseAmplitude(arc, semimajoraxis, k, n):
    value = 0
    amp = 0
    d = 2 * numpy.pi / n
    while value < arc / semimajoraxis:
        trig = numpy.sin(amp)
        value += numpy.sqrt(1-k*trig*trig) * d
        amp += d
    return amp