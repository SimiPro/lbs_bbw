#pragma once

#include <ostream>
#include <cmath>
#include <cassert>

template <class Value, class Deriv>
class AutoDiff
{
public:
	AutoDiff() {
	}

	template<class T>
	AutoDiff(const T &c) : m_x((Value)c), m_d((Deriv)0)	{
	}

	AutoDiff(const Value &x, const Deriv &d) : m_x(x), m_d(d) {
	}

	bool operator==(const AutoDiff<Value, Deriv> &other) const {
		return other.value() == this->value();
	}

	bool operator!=(const AutoDiff<Value, Deriv> &other) const {
		return other.value() != this->value();
	}

	AutoDiff<Value, Deriv> operator+(const AutoDiff<Value, Deriv> &other) const {
		return AutoDiff<Value, Deriv>(m_x + other.m_x, m_d + other.m_d);
	}

	AutoDiff<Value, Deriv> operator-(const AutoDiff<Value, Deriv> &other) const {
		return AutoDiff<Value, Deriv>(m_x - other.m_x, m_d - other.m_d);
	}

	AutoDiff<Value, Deriv> operator-() const {
		return AutoDiff<Value, Deriv>(-m_x, -m_d);
	}

	AutoDiff<Value, Deriv> operator*(const AutoDiff<Value, Deriv> &other) const {
		// D(x*y) = x*dy + dx*y
		return AutoDiff<Value, Deriv>(m_x * other.m_x, other.m_d * m_x + m_d * other.m_x);
	}

	AutoDiff<Value, Deriv> operator/(const AutoDiff<Value, Deriv> &other) const {
		// D(x/y) = (dx*y - x*dy) / y*y
		Value x2 = other.m_x * other.m_x;
		return AutoDiff<Value, Deriv>(m_x / other.m_x, m_d*(other.m_x/x2) - other.m_d*(m_x/x2));
	}

	AutoDiff<Value, Deriv> &operator+=(const AutoDiff<Value, Deriv> &other) {
		m_x = m_x + other.m_x;
		m_d = m_d + other.m_d;
		return *this;
	}

	AutoDiff<Value, Deriv> &operator-=(const AutoDiff<Value, Deriv> &other) {
		m_x = m_x - other.m_x;
		m_d = m_d - other.m_d;

		return *this;
	}

	AutoDiff<Value, Deriv> &operator*=(const AutoDiff<Value, Deriv> &other) {
		*this = *this * other;
		return *this;
	}

	AutoDiff<Value, Deriv> &operator/=(const AutoDiff<Value, Deriv> &other) {
		*this = *this / other;
		return *this;
	}

	bool operator>(const AutoDiff<Value, Deriv> &d) const {
		return m_x > d.m_x;
	}

	bool operator<(const AutoDiff<Value, Deriv> &d) const {
		return m_x < d.m_x;
	}

	bool operator>=(const AutoDiff<Value, Deriv> &d) const {
		return m_x >= d.m_x;
	}

	bool operator<=(const AutoDiff<Value, Deriv> &d) const {
		return m_x <= d.m_x;
	}

	const Value &value() const { return m_x; }
	Value &value() { return m_x; }

	const Deriv &deriv() const { return m_d; }
	Deriv &deriv() { return m_d; }

private:
	Value m_x;			// value
	Deriv m_d;			// derivative
};

template<class Value, class Deriv>
AutoDiff<Value, Deriv> operator+(const Value &a, const AutoDiff<Value, Deriv> &y)
{
	// d(a+y) = dy/dx
	return AutoDiff<Value, Deriv>(a + y.value(), y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> operator-(const Value &a, const AutoDiff<Value, Deriv> &y)
{
	// d(a-y) = -dy/dx
	return AutoDiff<Value, Deriv>(a - y.value(), -y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> operator*(const Value &a, const AutoDiff<Value, Deriv> &y)
{
	// d(a*y)/dx = a*dy/dx
	return AutoDiff<Value, Deriv>(a * y.value(), a * y.deriv());
}

// TODO: test this!
template<class S, class Value, class Deriv>
AutoDiff<Value, Deriv> operator*(const S &a, const AutoDiff<Value, Deriv> &y)
{
	// d(a*y)/dx = a*dy/dx
	return AutoDiff<Value, Deriv>(a * y.value(), a * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> operator/(const Value &a, const AutoDiff<Value, Deriv> &y)
{
	// D(a/y) = -a/y^2 * dy/dx
	return AutoDiff<Value, Deriv>(a / y.value(), -a / (y.value() * y.value()) * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> sin(const AutoDiff<Value, Deriv> &y)
{
	// d(sin(y))/dx = cos(y) * dy/dx
	return AutoDiff<Value, Deriv>(sin(y.value()), cos(y.value()) * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> cos(const AutoDiff<Value, Deriv> &y)
{
	// d(cos(y))/dx = -sin(y) * dy/dx
	return AutoDiff<Value, Deriv>(cos(y.value()), -sin(y.value()) * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> tan(const AutoDiff<Value, Deriv> &y)
{
	Value tanValue = tan(y.value());

	// d(tan(y))/dx = (1 + tan(y)^2) * dy/dx
	return AutoDiff<Value, Deriv>(tanValue, ((Value)1 + tanValue*tanValue) * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> acos(const AutoDiff<Value, Deriv> &y)
{
	// d(acos(x))/dx = -1/sqrt(1-y*y) * dy/dx
	return AutoDiff<Value, Deriv>(acos(y.value()), (Value(-1)/sqrt(Value(1)-y.value()*y.value())) * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> sqrt(const AutoDiff<Value, Deriv> &y)
{
	// d(sqrt(y))/dx = 1/2 * 1/sqrt(y) * dy/dx
	return AutoDiff<Value, Deriv>(sqrt(y.value()), Value(1)/Value(2) * Value(1)/sqrt(y.value()) * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> log(const AutoDiff<Value, Deriv> &y)
{
	// d(log(y))/dx = 1.0 / y * dy/dx
	return AutoDiff<Value, Deriv>(log(y.value()), Value(1)/y.value() * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> pow(const AutoDiff<Value, Deriv> &y, const double &a)
{
	// d(y^a)/dx = a*y^{a-1} * dy/dx
	return AutoDiff<Value, Deriv>(pow(y.value(), a), a*pow(y.value(), a-1) * y.deriv());
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> pow(const AutoDiff<Value, Deriv> &y1, const AutoDiff<Value, Deriv> &y2)
{
	// D(y1^y2) = y1^y2 * (dy1/dx*ln(y2) + (y2*dy1/dx)/y1)
	return AutoDiff<Value, Deriv>(pow(y1.value(), y2.value()),
								  pow(y1.value(), y2.value()) * (y2.deriv()*log(y1.value()) + y2.value()*y1.deriv()/y1.value()));
}

template<class Value, class Deriv>
AutoDiff<Value, Deriv> fabs(const AutoDiff<Value, Deriv> &s)
{
	if(s.value() >= 0)
		return s;
	else
		return -s;
}

template<class Value, class Deriv>
std::ostream& operator<<(std::ostream& stream, const AutoDiff<Value, Deriv> &s) {
	stream << s.value() << "(" << s.deriv() << ")";
	return stream;
}
