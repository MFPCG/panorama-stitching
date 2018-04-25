CC := g++
CFLAGS := -c
OBJECTS := mystitch.o camera.o Correcting.o
TARGET := panostitch
$(TARGET) : $(OBJECTS)
	$(CC)  -o $(TARGET) $(OBJECTS) `pkg-config --cflags --libs opencv`
mystitch.o : mystitch.cpp camera.h Correcting.h
camera.o : camera.cpp
Correcting.o : Correcting.cpp
clean : 
	rm $(OBJECTS) $(TARGET)
