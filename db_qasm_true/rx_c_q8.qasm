OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
rx(pi) q[0];
rx(pi/2) q[1];
rx(pi/4) q[2];
rx(pi/8) q[3];
rx(pi/16) q[4];
rx(pi/32) q[5];
rx(pi/64) q[6];
rx(pi/128) q[7];