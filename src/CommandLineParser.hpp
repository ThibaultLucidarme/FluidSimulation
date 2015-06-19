/*
 * minimal code
 *
 *#include "CommandLineParser.hpp"
    p::CommandLineParser parser(argc, argv);

    int input = parser.addOption<int>("-i",-17,"test int");
    std::string allo = parser.addOption<std::string>("-s","coucou","test string");
    std::string peep = parser.addOption<std::string>("-p","salut");


 */
#include <iostream>
#include <sstream>
#include <string>
#include <cstdarg>
#include <vector>
#include <exception>
#include <map>
#include "convertType.hpp"

namespace p
{
class CommandLineParser
{
private:
int								   _numArg;
std::vector<std::string>		   _argList;
std::map<std::string, std::string> _argmap;
std::map<std::string, std::string> _descriptionmap;
bool _displayHelp;
char* _appName;

public:
CommandLineParser(int argc, char** argv)
{
	_numArg = argc;
	_displayHelp = false;
	_appName = argv[0];

	// convert char** to vector<string>

	for (int i = 1; i < argc; i++)
	{
		std::stringstream ss;
		std::string		  s;
		ss << argv[i];
		ss >> s;
		_argList.push_back(s);
		
		if(s=="-h" || s=="--help")
		{
			_displayHelp = true;
		}
	}

	// map <option , Argument>

	for (unsigned int i = 0; i < _argList.size(); i++)
	{
		if (_argList[i][0] == '-')
		{
			if(i!=_argList.size()-1) // _argList[i + 1] exists
			{
				if (_argList[i + 1][0] != '-')
					_argmap.insert(std::pair<std::string, std::string>(_argList[i], _argList[i + 1]) );
				else
					_argmap.insert(std::pair<std::string, std::string>(_argList[i],  "") );
			}
			else
			{
				_argmap.insert(std::pair<std::string, std::string>(_argList[i],  "") );
			}
		}
	}
}

template<typename Type>
Type addOption(std::string optName, Type defaultValue, std::string description = "default description")
{
	Type result;

	//if option is found

	if (_argmap.find(optName) != _argmap.end() )
		p::ConvertType(_argmap.find(optName)->second, result);

	//if option not found, assign default value

	else
		result = defaultValue;

	std::string defaultVal;
	p::ConvertType(defaultValue, defaultVal);
	description = description+" [default:"+defaultVal+"]";
	_descriptionmap.insert(std::pair<std::string, std::string>(optName, description) );

	return result;
}

// Needs to be added at the end of all calls
void addHelpOption(std::string s="")
{
	if(_displayHelp)
	{
		std::cout<<std::endl<<"Usage:"<<std::endl;
		std::cout<<_appName<<" [options]"<<std::endl;
		std::cout<<"Options:"<<std::endl;
		for (auto& opt: _descriptionmap) 
		{
			std::cout <<"\t"<< opt.first << ":\t" << opt.second << '\n';
		}
		
		std::cout << std::endl<<s<<std::endl;
		
		exit(EXIT_SUCCESS);
		
	}
}



};


/*
 *
 * CommandLineParser parser(argc, argv);
 * parser.addOption<int>( "-i", &input);
 * parser.addOption<std::string>( "-s", &allo, &quiEst, &la );
 * parser.addHelpOption();
 *
 *
 *** overload
 *
 *
 * int i = parser.addOption<int>("-i", defaultValue);
 * std::string s[] = parser.addOption<std::string>( "-s", "defaultValue" );
 * parser.addHelpOption();
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 * */
}
