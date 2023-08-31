	#include <stdio.h>
	#include <stdbool.h>

	void recebeVetor(double vetor[], int tamanhoVetor){
		for (int i = 0; i < tamanhoVetor; i++) { // Inicializa vetor
			vetor[i] = 0;
		}
		for (int i = 0; i < tamanhoVetor; i++) {
			scanf("%lf", &vetor[i]);
		}
		
	}

	void criaMatriz(double vetor[], int m, int n, double matriz[m][n], int posicaoUtilizada[m][n]) {
		int contador = 0;
		for (int i = m - 1, j = n - 1, direction = 0; contador < m * n;) {
			//directions: 0 -> north; 1 -> west; 2 -> south; 3 -> east
			matriz[i][j] = vetor[contador];
			posicaoUtilizada[i][j] = 1;
			if (contador == m * n - 1) {
				//printf("finish");
				contador = m * n;
			} else if (direction == 0) {
				if (i - 1 >= 0) {
					if (posicaoUtilizada[i-1][j] == 0) {
						i -= 1;
						contador++;
					} else {
						direction = 1;
					}
				} else {
					direction = 1;
				}
			} else if (direction == 1) {
				if (j - 1 >= 0) {
					if (posicaoUtilizada[i][j - 1] == 0) {
						j -= 1;
						contador++;
					} else {
						direction = 2;
					}
				} else {
					direction = 2;
				}
				
			} else if (direction == 2) {
				if (i + 1 <= m - 1) {
					if(posicaoUtilizada[i+1][j] == 0) {
						i += 1;
						contador++;
					} else {
						direction = 3;
					}
				} else {
					direction = 3;
				}
			} else if (direction == 3) {
				if (j + 1 <= n - 1) {
					if(posicaoUtilizada[i][j+1] == 0) {
						j += 1;
						contador++;
					} else {
						direction = 0;
					}
				} else {
					direction = 0;
				}
			}
			/*  ferramenta pra debug
			printf("contador: %d", contador);
			printf("\n");
			printf("vetor[contador] %lf", vetor[contador]);
			printf("\n");
			printf("direction: %d", direction);
			printf("\n");
			printf("i: %d", i);
			printf("\n");
			printf("j: %d", j); 
			printf("\n");
			printf("posicaoUtilizada[i-1][j]: %d", posicaoUtilizada[i-1][j]);
			printf("\n");
			*/
		}
	
	}

	void imprimeMatriz(int m, int n, double matriz[][n]){
		int i, j;
		for (i = 0; i < m; i++) {
			for(j = 0; j < n; j++) {
				if (j < n - 1) {
					printf("%.1f ", matriz[i][j]);
				} else {
					printf("%.1f", matriz[i][j]);
				}
			}
			printf("\n");
		}
	}
	
	void inicializaPosicaoUtilizada(int m, int n, int posicaoUtilizada[][n]) {
		for (int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				posicaoUtilizada[i][j] = 0;
			}
		}
	}

	int main() {
		int m = 0, n = 0;
		scanf("%d %d", &m, &n);
		double vetor[m * n];
		for (int i = 0; ) {
		}
		
		return 0;
	}
