CC=nvcc
CFLAGS=-arch=sm_12 
LDFLAGS=

.SUFFIXES: .cu

SRC=$(PREFIX).cu
SRC_MED=$(PREFIX)_med.cu
SRC_ADV=$(PREFIX)_adv.cu

OBJ=$(SRC:.cu=.o)
OBJ_MED=$(SRC_MED:.cu=.o)
OBJ_ADV=$(SRC_ADV:.cu=.o)

all: $(PREFIX) $(PREFIX)_med $(PREFIX)_adv

$(PREFIX): $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) -o $@

$(PREFIX)_med: $(OBJ_MED)
	$(CC) $(LDFLAGS) $(OBJ_MED) -o $@

$(PREFIX)_adv: $(OBJ_ADV)
	$(CC) $(LDFLAGS) $(OBJ_ADV) -o $@

.cu.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(PREFIX) $(PREFIX)_med $(PREFIX)_adv *.o
