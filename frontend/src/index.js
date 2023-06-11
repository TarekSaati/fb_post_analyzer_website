import React, {useState} from "react";
import {
    Button,
    Flex,
    Input,
    InputGroup,
    InputLeftElement,
    Modal,
    ModalBody,
    ModalCloseButton,
    ModalContent,
    ModalFooter,
    ModalHeader,
    ModalOverlay,
    Stack,
    Heading,
    useDisclosure,
    ChakraProvider
} from "@chakra-ui/react";
import { render } from 'react-dom';
import { AiFillHome, AiOutlineUserAdd } from "react-icons/ai";
import Postlist from "./Components/Postlist";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";

const TokenContext = React.createContext({token: '', setToken: () => {}})

function Login() {
  const {setToken} = React.useContext(TokenContext)
  const {isOpen, onOpen, onClose} = useDisclosure()
  const [email, setEmail] = React.useState('')
  const [password, setPassword] = React.useState('')

  const loginUser = async () => {  
    let formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    const ret = await fetch(`http://localhost:8000/login/`, {method: "POST", body: formData})
    const token_data = await ret.json()
    setToken(token_data.access_token)
    onClose()
  }

  return (
    <>
      <Button leftIcon={<AiFillHome />} colorScheme='cyan' variant='solid' onClick={onOpen}>Login</Button>
        <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay/>
          <ModalContent>
            <ModalHeader>Enter Login Credentials</ModalHeader>
            <ModalCloseButton/>
            <ModalBody>
              <Stack spacing={4}>
              <InputGroup size="md">
              <InputLeftElement pointerEvents='none'>
              </InputLeftElement>
                <Input
                  pr="4.5rem"
                  type="email"
                  placeholder="email"
                  aria-label="email"
                  onChange={e => setEmail(e.target.value)}
                />
                </InputGroup>
                <InputGroup size="md">
                <InputLeftElement pointerEvents='none'>
                </InputLeftElement>
                <Input
                  pr="4.5rem"
                  type="password"
                  placeholder="password"
                  aria-label="password"
                  onChange={e => setPassword(e.target.value)}
                />
                </InputGroup>
                </Stack>
            </ModalBody>

            <ModalFooter>
              <Button h="1.5rem" size="sm" onClick={loginUser}>Login</Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
    </>
  )
}

function Register() {
  const {isOpen, onOpen, onClose} = useDisclosure()
  const [email, setEmail] = React.useState('')
  const [password, setPassword] = React.useState('')
  const [firstname, setFirstname] = React.useState('')
  const [lastname, setLastname] = React.useState('')

  const registerUser = async () => {
    const user_info = {
      'email': email,
      'firstName': firstname,
      'lastName': lastname,
      'password': password
    }
    const ret = await fetch("http://localhost:8000/users/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(user_info)
    })
    const ret_email = await ret.json().email
    alert(`User ${firstname} was registered successfully!`)
    onClose()
  }

  return (
    <>
      <Button rightIcon={<AiOutlineUserAdd />} colorScheme='pink' variant='outline' onClick={onOpen}>Sign Up</Button>
        <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay/>
          <ModalContent>
            <ModalHeader>Enter User Info</ModalHeader>
            <ModalCloseButton/>
            <ModalBody>
              <Stack>
              <InputGroup size="md">
              <InputLeftElement pointerEvents='none'>
              </InputLeftElement>
                <Input
                  pr="4.5rem"
                  type="text"
                  placeholder="first name"
                  aria-label="first name"
                  onChange={e => setFirstname(e.target.value)}
                />
                </InputGroup>
                <InputGroup size="md">
                <InputLeftElement pointerEvents='none'>
                </InputLeftElement>
                <Input
                  pr="4.5rem"
                  type="text"
                  placeholder="last name"
                  aria-label="last name"
                  onChange={e => setLastname(e.target.value)}
                />
                </InputGroup>
                <InputGroup size="md">
                <InputLeftElement pointerEvents='none'>
                </InputLeftElement>
                <Input
                  pr="4.5rem"
                  type="email"
                  placeholder="email"
                  aria-label="email"
                  onChange={e => setEmail(e.target.value)}
                />
                </InputGroup>
                <InputGroup size="md">
                <InputLeftElement pointerEvents='none'>
                </InputLeftElement>
                <Input
                  pr="4.5rem"
                  type="password"
                  placeholder="password"
                  aria-label="password"
                  onChange={e => setPassword(e.target.value)}
                />
              </InputGroup>
              </Stack>
            </ModalBody>

            <ModalFooter>
              <Button h="1.5rem" size="sm" onClick={registerUser}>Register</Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
    </>
  )
}

const Header = () => {
  return (
      <Flex
          as="nav"
          align="center"
          justify="space-between"
          wrap="wrap"
          padding="0.5rem"
          bg="gray.400"
      >
        <Flex align="center" mr={5}>
          <Heading as="h2" size='md'>Welcome to Facebook Arabic Posts Analyzer</Heading>
        </Flex>

        <Stack direction='row' spacing={4}>
        <InputGroup>
          <Login />
          </InputGroup>
          </Stack>

            <Stack direction='row' spacing={4}>
            <InputGroup>
            <Register />
            </InputGroup>
            </Stack>
        
      </Flex>
  );
};


function App() {
  const [token, setToken] = useState('')
  return (
    <Router>
      <ChakraProvider>
        <TokenContext.Provider value={{token, setToken}}>
          <Header />
          <Link to='/posts'>Show Posts by Topic</Link>
          <Routes>
            <Route path='/posts' element={<Postlist token={token}/>}></Route>
          </Routes>
        </TokenContext.Provider>
      </ChakraProvider>
    </Router>
  )
}

const rootElement = document.getElementById("root")
render(<App />, rootElement)
