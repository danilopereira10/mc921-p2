###############################################################################################
######################### Compilação com suporte a múltiplos arquivos #########################
###############################################################################################

.PHONY: build clean

LAB = lab01

OBJECTS = $(patsubst %.c, %.o, $(wildcard *.c))
GFLAGS  = -std=c99 -Wall -Werror

build: $(OBJECTS)
	gcc $(GFLAGS) *.o -o $(LAB) -lm

%.o: %.c
	gcc $(GFLAGS) -g -c $< -o $@

clean:
	rm -f *.o


###############################################################################################
####################### Suporte para testar o programa usando os testes #######################
###############################################################################################

.PHONY: testar_abertos testar_fechados

testar_abertos:
	@set -e ; \
	if [ ! -d testes_abertos ] ; then \
		echo "\033[1;31mDiretório 'testes_abertos' não encontrado!\033[0m" ; \
		exit 1 ; \
	fi ; \
	for entrada in testes_abertos/*.in ; do \
		echo "\033[1;34mTestando \033[4;34m$$entrada\033[0m" ; \
		saida=$$(echo "$$entrada" | sed -re 's/\.in$$/.out/' ) ; \
		resposta=$$(echo "$$entrada" | sed -re 's/\.in$$/.res/' ) ; \
		./$(LAB) < "$$entrada" > "$$saida" ; \
		if diff "$$resposta" "$$saida" ;  then \
			echo "\033[1;32mResultado correto!\033[0m" ; \
		else \
			echo "\033[1;31mFalhou!\033[0m" ; \
			exit 1; \
		fi ; \
		echo ; \
	done

testar_fechados:
	@set -e ; \
	if [ ! -d testes_fechados ] ; then \
		echo "\033[1;31mDiretório 'testes_fechados' não encontrado!\033[0m" ; \
		exit 1 ; \
	fi ; \
	for entrada in testes_fechados/*.in ; do \
		echo "\033[1;34mTestando \033[4;34m$$entrada\033[0m" ; \
		saida=$$(echo "$$entrada" | sed -re 's/\.in$$/.out/' ) ; \
		resposta=$$(echo "$$entrada" | sed -re 's/\.in$$/.res/' ) ; \
		./$(LAB) < "$$entrada" > "$$saida" ; \
		if diff "$$resposta" "$$saida" ;  then \
			echo "\033[1;32mResultado correto!\033[0m" ; \
		else \
			echo "\033[1;31mFalhou!\033[0m" ; \
			exit 1; \
		fi ; \
		echo ; \
	done


###############################################################################################
###################### Suporte para checar o programa usando o valgrind #######################
###############################################################################################

.PHONY: checar_abertos checar_fechados

checar_abertos:
	@set -e ; \
	if [ ! -d testes_abertos ] ; then \
		echo "\033[1;31mDiretório 'testes_abertos' não encontrado!\033[0m" ; \
		exit 1 ; \
	fi ; \
	for entrada in testes_abertos/*.in ; do \
		echo "\033[1;34mTestando \033[4;34m$$entrada\033[0m" ; \
		saida=$$(echo "$$entrada" | sed -re 's/\.in$$/.out/' ) ; \
		resposta=$$(echo "$$entrada" | sed -re 's/\.in$$/.res/' ) ; \
		if valgrind -q --leak-check=full ./$(LAB) < "$$entrada" > "$$saida" ;  then \
			echo "\033[1;32mResultado correto!\033[0m" ; \
		else \
			echo "\033[1;31mFalhou!\033[0m" ; \
			exit 1; \
		fi ; \
		echo ; \
	done

checar_fechados:
	@set -e ; \
	if [ ! -d testes_fechados ] ; then \
		echo "\033[1;31mDiretório 'testes_fechados' não encontrado!\033[0m" ; \
		exit 1 ; \
	fi ; \
	for entrada in testes_fechados/*.in ; do \
		echo "\033[1;34mTestando \033[4;34m$$entrada\033[0m" ; \
		saida=$$(echo "$$entrada" | sed -re 's/\.in$$/.out/' ) ; \
		resposta=$$(echo "$$entrada" | sed -re 's/\.in$$/.res/' ) ; \
		if valgrind -q --leak-check=full ./$(LAB) < "$$entrada" > "$$saida" ;  then \
			echo "\033[1;32mResultado correto!\033[0m" ; \
		else \
			echo "\033[1;31mFalhou!\033[0m" ; \
			exit 1; \
		fi ; \
		echo ; \
	done


###############################################################################################
################################ Suporte para baixar os testes ################################
###############################################################################################

.PHONY: baixar_abertos baixar_fechados

baixar_abertos:
	@mkdir -p testes_abertos ; \
	sbx=AmbienteDeTeste ; \
	lnk=https://susy.ic.unicamp.br:9999/mc202e ; \
	lab=$$(echo "$(LAB)" | sed -re 's/lab//') ; \
	#if [ -f testes_abertos/arq15.res ] ; then \
	#	echo "Testes abertos já foram baixados." ; \
	#	exit 0 ; \
	#fi ; \
	curl -sk "$$lnk/Lab$$lab-$$sbx/dados/arq15.in" | grep -e '<HTML>' ; \
	if [ $$?  = "0" ] ; then \
		echo "\033[1;31mCertifique-se de que os testes abertos estejam liberados\033[0m" ; \
		exit 1 ; \
	fi ; \
	for i in $(shell seq -f "%02g" 1 15) ; do \
		echo "\033[1;34mBaixando arquivos de teste aberto #$$i\033[0m" ; \
		for ext in .in .res ; do \
			url=$$lnk/Lab$$lab-$$sbx/dados/arq$$i$$ext ; \
			echo "Lab$$lab-$$sbx/dados/arq$$i$$ext" ; \
			curl -s -k "$$url" > testes_abertos/arq$$i$$ext ; \
		done ; \
		echo "\033[1;32mConcluído!\033[0m" ; \
		echo ; \
	done

baixar_fechados:
	@mkdir -p testes_fechados ; \
	sbx=AmbienteDeTeste ; \
	lnk=https://susy.ic.unicamp.br:9999/mc202e ; \
	lab=$$(echo "$(LAB)" | sed -re 's/lab//') ; \
	#if [ -f testes_fechados/arq25.res ] ; then \
	#	echo "Testes fechados já foram baixados." ; \
	#	exit 0 ; \
	#fi ; \
	curl -sk "$$lnk/Lab$$lab-$$sbx/dados/arq25.in" | grep -e '<HTML>' ; \
	if [ $$?  = "0" ] ; then \
		echo "\033[1;31mCertifique-se de que os testes fechados estejam liberados\033[0m" ; \
		exit 1 ; \
	fi ; \
	for i in $(shell seq 16 25) ; do \
		echo "\033[1;34mBaixando arquivos de teste fechado #$$i\033[0m" ; \
		for ext in .in .res ; do \
			url=$$lnk/Lab$$lab-$$sbx/dados/arq$$i$$ext ; \
			echo "Lab$$lab-$$sbx/dados/arq$$i$$ext" ; \
			curl -s -k "$$url" > testes_fechados/arq$$i$$ext ; \
		done ; \
		echo "\033[1;32mConcluído!\033[0m" ; \
		echo ; \
	done
