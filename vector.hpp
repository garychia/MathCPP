namespace Math
{
    template <class T>
    class Vector3D
    {
    public:
        T x;
        T y;
        T z;

        Vector3D(T x = 0, T y = 0, T z = 0) : x(x), y(y), z(z) {}

        template<class OtherType>
        Vector3D Add(const Vector3D<OtherType>& other) const {
            return Vector3D(x + other.x, y + other.y, z + other.z);
        }

        template<class OtherType>
        Vector3D operator+(const Vector3D<OtherType>& other) const {
            return this->Add(other);
        }

        template<class OtherType>
        Vector3D& operator+=(const Vector3D<OtherType>& other) {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }

        template<class OtherType>
        Vector3D Minus(const Vector3D<OtherType>& other) const {
            return Vector3D(x - other.x, y - other.y, z - other.z);
        }

        template<class OtherType>
        Vector3D operator-(const Vector3D<OtherType>& other) const {
            return this->Minus(other);
        }

        template<class OtherType>
        Vector3D& operator-=(const Vector3D<OtherType>& other) {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }

        template<class OtherType>
        Vector3D Scale(const OtherType& scaler) const {
            return Vector3D(x * scaler, y * scaler, z * scaler);
        }

        template<class OtherType>
        Vector3D operator*(const OtherType& scaler) const {
            return this->Scale(scaler);
        }

        template<class OtherType>
        Vector3D Divide(const OtherType& scaler) const {
            return Vector3D(x / scaler, y / scaler, z / scaler);
        }

        template<class OtherType>
        Vector3D operator/(const OtherType& scaler) const {
            return this->Divide(scaler);
        }

        template<class OtherType>
        Vector3D operator/=(const OtherType& scaler) const {
            x /= scaler;
            y /= scaler;
            z /= scaler;
            return *this;
        }

        template<class OtherType>
        Vector3D Dot(const Vector3D<OtherType>& other) const {
            return Vector3D(x * other.x, y * other.y, z * other.z);
        }

        template<class OtherType>
        Vector3D Cross(const Vector3D<OtherType>& other) const {
            return Vector3D(y * other.z - other.y * z, z * other.x - other.z * x, x * other.y - other.x * y);
        }
    };
}