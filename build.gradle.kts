plugins {
    id("application")
}

group = "br.com.usp.ach2016"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.ejml:ejml-simple:0.44.0")
}