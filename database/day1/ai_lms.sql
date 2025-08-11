-- 공통 스키마
CREATE SCHEMA IF NOT EXISTS shared_lms;

-- 서울캠퍼스
CREATE SCHEMA IF NOT EXISTS seoul_lms;

-- 제주캠퍼스
CREATE SCHEMA IF NOT EXISTS jeju_lms;

-- Course
CREATE TABLE shared_lms.course (
  course_id SERIAL PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  description VARCHAR(1000),
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  status VARCHAR(50),
  category VARCHAR(100)
);

-- CourseDescription
CREATE TABLE shared_lms.coursedescription (
  description_id SERIAL PRIMARY KEY,
  course_id INT NOT NULL REFERENCES shared_lms.course(course_id) ON DELETE CASCADE,
  content TEXT NOT NULL
);

-- Instructor
CREATE TABLE shared_lms.instructor (
  instructor_id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  specialty VARCHAR(255),
  bio TEXT
);

-- TeachingAssignment
CREATE TABLE shared_lms.teaching_assignment (
  assignment_id SERIAL PRIMARY KEY,
  instructor_id INT NOT NULL REFERENCES shared_lms.instructor(instructor_id) ON DELETE CASCADE,
  course_id INT NOT NULL REFERENCES shared_lms.course(course_id) ON DELETE CASCADE,
  assigned_at DATE DEFAULT CURRENT_DATE
);

-- Student
CREATE TABLE seoul_lms.student (
  student_id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  phone VARCHAR(20),
  registration_date DATE DEFAULT CURRENT_DATE
);

-- Enrollment
CREATE TABLE seoul_lms.enrollment (
  enrollment_id SERIAL PRIMARY KEY,
  student_id INT NOT NULL REFERENCES seoul_lms.student(student_id) ON DELETE CASCADE,
  course_id INT NOT NULL REFERENCES shared_lms.course(course_id) ON DELETE CASCADE,
  enrolled_at DATE DEFAULT CURRENT_DATE,
  progress FLOAT CHECK (progress >= 0 AND progress <= 100),
  completion_status VARCHAR(50)
);

-- Review
CREATE TABLE seoul_lms.review (
  review_id SERIAL PRIMARY KEY,
  student_id INT NOT NULL REFERENCES seoul_lms.student(student_id) ON DELETE CASCADE,
  course_id INT NOT NULL REFERENCES shared_lms.course(course_id) ON DELETE CASCADE,
  rating INT CHECK (rating BETWEEN 1 AND 5),
  comment TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embedding (VECTOR type requires pgvector extension)
CREATE TABLE seoul_lms.embedding (
  embedding_id SERIAL PRIMARY KEY,
  student_id INT NOT NULL REFERENCES seoul_lms.student(student_id) ON DELETE CASCADE,
  embedding VECTOR(4),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- InstructorAssignment
CREATE TABLE seoul_lms.instructor_assignment (
  id SERIAL PRIMARY KEY,
  instructor_id INT NOT NULL REFERENCES shared_lms.instructor(instructor_id) ON DELETE CASCADE,
  assigned_at DATE DEFAULT CURRENT_DATE
);

-- Student
CREATE TABLE jeju_lms.student (
  student_id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  phone VARCHAR(20),
  registration_date DATE DEFAULT CURRENT_DATE
);

-- Enrollment
CREATE TABLE jeju_lms.enrollment (
  enrollment_id SERIAL PRIMARY KEY,
  student_id INT NOT NULL REFERENCES jeju_lms.student(student_id) ON DELETE CASCADE,
  course_id INT NOT NULL REFERENCES shared_lms.course(course_id) ON DELETE CASCADE,
  enrolled_at DATE DEFAULT CURRENT_DATE,
  progress FLOAT CHECK (progress >= 0 AND progress <= 100),
  completion_status VARCHAR(50)
);

-- Review
CREATE TABLE jeju_lms.review (
  review_id SERIAL PRIMARY KEY,
  student_id INT NOT NULL REFERENCES jeju_lms.student(student_id) ON DELETE CASCADE,
  course_id INT NOT NULL REFERENCES shared_lms.course(course_id) ON DELETE CASCADE,
  rating INT CHECK (rating BETWEEN 1 AND 5),
  comment TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embedding
CREATE TABLE jeju_lms.embedding (
  embedding_id SERIAL PRIMARY KEY,
  student_id INT NOT NULL REFERENCES jeju_lms.student(student_id) ON DELETE CASCADE,
  embedding VECTOR(4),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- InstructorAssignment
CREATE TABLE jeju_lms.instructor_assignment (
  id SERIAL PRIMARY KEY,
  instructor_id INT NOT NULL REFERENCES shared_lms.instructor(instructor_id) ON DELETE CASCADE,
  assigned_at DATE DEFAULT CURRENT_DATE
);
