OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rx(pi) q[0];
rx(pi/2) q[1];
rx(pi/4) q[2];
rx(pi/8) q[3];